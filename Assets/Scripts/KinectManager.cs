using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Windows.Kinect;
using System.Collections.Generic;
using UnityEngine;
using System.Drawing;
using System.Linq;

public class KinectManager : MonoBehaviour {

    KinectSensor _sensor;
    DepthFrameReader depthFrameReader;
    const int WIDTH = 512;
    const int HEIGHT = 424;
    const int MIN_DEPTH = 500;
    const int MAX_DEPTH = 1300;
    const int DEPTH_TOL = 200;
    private int beforeCenterDepth = 0;
    public bool confirmacion = false;
	public Transform ball;

    List<System.Drawing.Point> trajectory = new List<System.Drawing.Point>();
    List<int> control = new List<int>();


    //Que tan grande se puede ver la pelotita en movimiento?
    int MIN_OBJECT_AREA = 50;
    int MAX_OBJECT_AREA = 1000;

    const short CONFIG_ITERACIONES = 30;

    //Estructuras
    ushort[] rawDepthPixels;
    int[] averageDepthConfig;
    List<int[]> listConfig;
    byte[] colorImage;

    Mat frameOpenCV;

    int depthPixel;
    int x = 0, y = 0;
    double area = 0;
    MCvScalar colorDetected;
    MCvScalar colorBounce;

    bool edgesDetected = false;
    int indexArea = 0;
    bool depthMapped = false;
    int minX = HEIGHT, maxX = 0, minY = WIDTH, maxY = 0;
    Emgu.CV.Util.VectorOfVectorOfPoint edgesTable = null;
    private int desborde = 0;
    private int beforeX;
    private int beforeY;

    DepthFrame frameeeee;



    List<Point> debugBounces = new List<Point>();

    public bool IsAvailable;


    public static KinectManager instance = null;


    void Awake()
    {
        if (instance == null)
        {
            instance = this;
        }
        else if (instance != this)
            Destroy(gameObject);
    }

    // Use this for initialization
    void Start()
    {

        System.GC.Collect();

        //Inicializo las variables
        string path = Application.dataPath + "\\Assets\\format.jpg";
        frameOpenCV = new Mat(path);

        rawDepthPixels = new ushort[WIDTH * HEIGHT];
        averageDepthConfig = new int[WIDTH * HEIGHT];
        listConfig = new List<int[]>();
        colorImage = new byte[WIDTH * HEIGHT * 3];


        colorDetected = new MCvScalar(0, 255, 0);
        colorBounce = new MCvScalar(0, 0, 255);

        debugBounces = new List<System.Drawing.Point>();

        _sensor = KinectSensor.GetDefault();

        if (_sensor != null)
        {
            IsAvailable = _sensor.IsAvailable;


            depthFrameReader = _sensor.DepthFrameSource.OpenReader();

            if (!_sensor.IsOpen)
            {
                _sensor.Open();
            }


        }
    }

    // Update is called once per frame
    void Update()
    {
        IsAvailable = _sensor.IsAvailable;

        if (depthFrameReader != null)
        {
            var frame = depthFrameReader.AcquireLatestFrame();

            if (frame != null) {

                frame.CopyFrameDataToArray(rawDepthPixels);
    
                //Primero acoto los limites de la mesa
                if (!edgesDetected)
                {
                    //Grafico las profundidades para detectar los bordes de la mesa
                    for (int depth = 0; depth < rawDepthPixels.Length; depth++)
                    {
                        depthPixel = rawDepthPixels[depth];
                        if (depthPixel > MIN_DEPTH && depthPixel < MAX_DEPTH)
                        {
                            colorImage[depth * 3] = 255;
                            colorImage[depth * 3 + 1] = 255;
                            colorImage[depth * 3 + 2] = 255;
                        }
                        else
                        {
							colorImage[depth * 3] = 0;
                            colorImage[depth * 3 + 1] = 0;
                            colorImage[depth * 3 + 2] = 0;
                        }
                    }
                    frameOpenCV.SetTo(colorImage);
                    UMat uimage = new UMat();
                    CvInvoke.CvtColor(frameOpenCV, uimage, ColorConversion.Bgr2Gray);
										
                    //Suavizo los puntos pequeños
                    Mat erodeElement = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new System.Drawing.Size(3, 3), new System.Drawing.Point(-1, -1));
                    Mat dilateElement = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new System.Drawing.Size(10, 10), new System.Drawing.Point(-1, -1));

                    MCvScalar scalarD = new MCvScalar(5, 5);
                    CvInvoke.Erode(uimage, uimage, erodeElement, new System.Drawing.Point(-1, -1), 4, BorderType.Constant, scalarD);
                    CvInvoke.Dilate(uimage, uimage, dilateElement, new System.Drawing.Point(-1, -1), 2, BorderType.Constant, scalarD);


                    //Busco contornos
                    edgesTable = new Emgu.CV.Util.VectorOfVectorOfPoint();
                    Mat heir = new Mat();
                    Image<Rgb, byte> imgout = new Image<Rgb, byte>(frameOpenCV.Width, frameOpenCV.Height, new Rgb(200, 200, 200));
                    CvInvoke.FindContours(uimage, edgesTable, heir, Emgu.CV.CvEnum.RetrType.Ccomp, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);

                    double maxArea = 0;
                    for (int i = 0; i < edgesTable.Size; i++)
                    {
                        var moment = CvInvoke.Moments(edgesTable[i]);
                        area = moment.M00;
                        //Me quedo con el area mas grande que es la mesa de ping pong
                        if (area > maxArea)
                        {
                            //PERO tengo que descartar el area que es todo el cuadrado(el frame de la imagen)
                            if (area < WIDTH * HEIGHT * 0.22 && area > WIDTH * HEIGHT * 0.18)
                            {
                                maxArea = area;
                                indexArea = i;
                            }

                        }

                    }

                    for (int i = 0; i < edgesTable[indexArea].Size; i++)
                    {
                        //Encuentro el X mas bajo y alto, lo mismo para la Y
                        if (edgesTable[indexArea][i].X > maxX)
                            maxX = edgesTable[indexArea][i].X + desborde;
                        if (edgesTable[indexArea][i].X < minX)
                            minX = edgesTable[indexArea][i].X - desborde;
                        if (edgesTable[indexArea][i].Y > maxY)
                            maxY = edgesTable[indexArea][i].Y + desborde;
                        if (edgesTable[indexArea][i].Y < minY)
                            minY = edgesTable[indexArea][i].Y - desborde;
                    }

                    CvInvoke.DrawContours(imgout, edgesTable, indexArea, new MCvScalar(255, 0, 0), 1);
                    CvInvoke.Circle(imgout, new System.Drawing.Point(minX, minY), 2, colorDetected, 2);
                    CvInvoke.Circle(imgout, new System.Drawing.Point(minX, maxY), 2, colorDetected, 2);
                    CvInvoke.Circle(imgout, new System.Drawing.Point(maxX, minY), 2, colorDetected, 2);
                    CvInvoke.Circle(imgout, new System.Drawing.Point(maxX, maxY), 2, colorDetected, 2);
                    edgesDetected = true;
					
					
                }


                //Despues mapeo la profundidad de la mesa
                if (!depthMapped && edgesDetected)
                {
                    //Cargo por unica vez la matriz de configuracion de profundidad
                    if (listConfig.Count < CONFIG_ITERACIONES)
                    {
                        var configDepth = new int[WIDTH * HEIGHT];
                        for (int row = minY; row < maxY; row++)
                        {
                            for (int col = minX; col < maxX; col++)
                            {
                                //transformo un fila columna en su equivalente de vector
                                depthPixel = rawDepthPixels[(row * WIDTH) + (col)];
                                if (depthPixel > MIN_DEPTH && depthPixel < MAX_DEPTH)
                                {
                                    configDepth[(row * WIDTH) + (col)] = depthPixel;
                                }
                                else
                                {
                                    //Le pongo 700 para que no se vaya a valor muy bajo con el -1 y no arruine el prom
                                    configDepth[(row * WIDTH) + (col)] = MAX_DEPTH - 200;
                                }

                            }
                        }

                        listConfig.Add(configDepth);
                        if (frame != null)
                        {
                            frame.Dispose();
                            frame = null;
                        }
                        return;
                    }


                    //Una vez que hizo las pasadas de configuracion saco el promedio
                    if (listConfig.Count == CONFIG_ITERACIONES)
                    {
                        //Saco el promedio para cada punto.
                        foreach (var item in listConfig)
                        {
                            for (int depth = 0; depth < averageDepthConfig.Length; depth++)
                            {
                                averageDepthConfig[depth] += item[depth];
                            }



                        }

                        for (int depth = 0; depth < averageDepthConfig.Length; depth++)
                        {
                            averageDepthConfig[depth] /= CONFIG_ITERACIONES;
                        }

                        depthMapped = true;
                        //Y limpio la matriz para que quede todo en negro.
                        for (int i = 0; i < colorImage.Length; i += 3)
                        {
                            colorImage[i + 0] = 0;
                            colorImage[i + 1] = 0;
                            colorImage[i + 2] = 0;
                        }



                    }

                }
                //Recien ahora puedo empezar a detectar profundidades y piques
                if (edgesDetected && depthMapped)
                {

                    for (int row = minY; row < maxY; row++)
                    {
                        for (int col = minX; col < maxX; col++)
                        {
                            //transformo un fila columna en su equivalente de vector
                            depthPixel = rawDepthPixels[(row * WIDTH) + (col)];
                            if (depthPixel > MIN_DEPTH && depthPixel < MAX_DEPTH && depthPixel < averageDepthConfig[(row * WIDTH) + (col)] - 5)
                            {
                                colorImage[(row * WIDTH * 3) + (col * 3) + 0] = 255;
                                colorImage[(row * WIDTH * 3) + (col * 3) + 1] = 255;
                                colorImage[(row * WIDTH * 3) + (col * 3) + 2] = 255;

                            }
                            else
                            {
                                colorImage[(row * WIDTH * 3) + (col * 3) + 0] = 0;
                                colorImage[(row * WIDTH * 3) + (col * 3) + 1] = 0;
                                colorImage[(row * WIDTH * 3) + (col * 3) + 2] = 0;
                            }
                        }
                    }

                    //Transformo mis pixeles en un formato OPENCV
                    frameOpenCV.SetTo(colorImage);
                    UMat uimage = new UMat();
                    CvInvoke.CvtColor(frameOpenCV, uimage, ColorConversion.Bgr2Gray);

                    //CvInvoke.Imshow("kinect camera", frameOpenCV);
                    //Suavizo los puntos pequeños
                    Mat erodeElement = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new System.Drawing.Size(3, 3), new System.Drawing.Point(-1, -1));
                    Mat dilateElement = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new System.Drawing.Size(5, 5), new System.Drawing.Point(-1, -1));

                    MCvScalar scalarD = new MCvScalar(5, 5);
                    CvInvoke.Erode(uimage, uimage, erodeElement, new System.Drawing.Point(-1, -1), 2, BorderType.Constant, scalarD);
                    CvInvoke.Dilate(uimage, uimage, dilateElement, new System.Drawing.Point(-1, -1), 4, BorderType.Constant, scalarD);

                    //CvInvoke.Imshow("Vision OPENCV", uimage);


                    //Busco contornos
                    Emgu.CV.Util.VectorOfVectorOfPoint countors = new Emgu.CV.Util.VectorOfVectorOfPoint();
                    Mat heir = new Mat();
                    Image<Rgb, byte> imgout = new Image<Rgb, byte>(frameOpenCV.Width, frameOpenCV.Height, new Rgb(200, 200, 200));
                    CvInvoke.FindContours(uimage, countors, heir, Emgu.CV.CvEnum.RetrType.Ccomp, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);



                    for (int i = 0; i < countors.Size; i++)
                    {
                        var moment = CvInvoke.Moments(countors[i]);
                        area = moment.M00;
                        //PENSAR ALGO MAS SELECTIVO QUE DESCARTE OBJETOS QUE NO SEAN CIRCULOS
                        if (area > MIN_OBJECT_AREA && area < MAX_OBJECT_AREA)
                        {
                            x = (int)(moment.M10 / area);
                            y = (int)(moment.M01 / area);
                            CvInvoke.DrawContours(imgout, countors, i, new MCvScalar(255, 0, 0), 1);
                            break;
                        }
                    }

                    if (x != 0 && y != 0)
                    {
                        int centerDepth = rawDepthPixels[y * WIDTH + x];


                        control.Add(centerDepth - averageDepthConfig[y * WIDTH + x]);

                        // AddTrajectory(false, new System.Drawing.Point(x, y));

                        //Es un pique solo si la diferencia entre la mesa y la pelota es minima, la mesa puede estar inclinada
                        if (centerDepth < averageDepthConfig[y * WIDTH + x] - 5 && centerDepth > averageDepthConfig[y * WIDTH + x] - DEPTH_TOL)
                        {
                            //Se detecto un pique
                            if (centerDepth - beforeCenterDepth >= 0)
                            {
                                if (centerDepth - beforeCenterDepth != 0)
                                    confirmacion = false;

                                System.Console.WriteLine("NO Pico" + " BF " + beforeCenterDepth + " CD " + centerDepth + " confirmacion: " + confirmacion);
                            }
                            else
                            {
                                if (!confirmacion)
                                {
                                    System.Console.WriteLine("Pico" + " BF " + beforeCenterDepth + " CD " + centerDepth + " confirmacion: " + confirmacion);
                                    debugBounces.Add(new System.Drawing.Point(beforeX, beforeY));
                                    confirmacion = true;
									


                                }

                            }
                            beforeCenterDepth = centerDepth;
                            beforeX = x;
                            beforeY = y;
							ball.position = new Vector3(beforeX+25,beforeY+25,0);

                        }

                        CvInvoke.Circle(imgout, new System.Drawing.Point(x, y), 20, colorDetected, 6);
                        CvInvoke.PutText(imgout, ((double)centerDepth / 1000).ToString("F") + "m", new System.Drawing.Point(x - 38, y + 50), FontFace.HersheyPlain, 1.3, colorBounce, 2);

                        x = 0;
                        y = 0;


                    }

                    foreach (var item in debugBounces)
                    {
                        CvInvoke.Circle(imgout, new System.Drawing.Point(item.X, item.Y), 10, colorBounce, 2);

                    }


                    if (debugBounces.Count > 1)
                        debugBounces.RemoveAt(0);
                    foreach (var item in trajectory)
                    {
                        CvInvoke.Circle(imgout, new System.Drawing.Point(item.X, item.Y), 10, colorBounce, 2);

                    }
                    CvInvoke.DrawContours(imgout, edgesTable, indexArea, new MCvScalar(255, 0, 0), 1);
                    CvInvoke.Imshow("Deteccion", imgout);

                }
                if (frame != null)
                {
                    frame.Dispose();
                    frame = null;
                }
            }

            if(frame != null)
            {
                frame.Dispose();
                frame = null;
            }


        }

    }



    void OnApplicationQuit()
    {
        if (depthFrameReader != null)
        {
            depthFrameReader.IsPaused = true;
            depthFrameReader.Dispose();
            depthFrameReader = null;
        }

        if (_sensor != null)
        {
            if (_sensor.IsOpen)
            {
                _sensor.Close();
            }

            _sensor = null;
        }
    }

}
