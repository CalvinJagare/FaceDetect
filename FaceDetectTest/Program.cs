using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Dnn;
using System;
using System.Drawing;
using System.IO;
using System.Linq;

class Program
{
    static void Main(string[] args)
    {
        
        string faceCascadeFile = @"D:\Code stuff\FaceDetect\FaceDetectTest\Resources\haarcascade_frontalface_default.xml"; // ADjust if needed
        if (!File.Exists(faceCascadeFile))
        {
            Console.WriteLine($"Error: The file '{faceCascadeFile}' does not exist.");
            return;
        }

        CascadeClassifier faceCascade;
        try
        {
            faceCascade = new CascadeClassifier(faceCascadeFile);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: Failed to load cascade classifier. {ex.Message}");
            return;
        }

        string modelFolder = @"D:\Code stuff\FaceDetect\FaceDetectTest\Models\";  // Adjust if needed
        string ageProto = Path.Combine(modelFolder, "age_deploy.prototxt");
        string ageModel = Path.Combine(modelFolder, "age_net.caffemodel");
        string genderProto = Path.Combine(modelFolder, "gender_deploy.prototxt");
        string genderModel = Path.Combine(modelFolder, "gender_net.caffemodel");

        if (!File.Exists(ageModel) || !File.Exists(ageProto) || !File.Exists(genderModel) || !File.Exists(genderProto))
        {
            Console.WriteLine("Error: One or more model files are missing!");
            return;
        }

        Net ageNet = DnnInvoke.ReadNetFromCaffe(ageProto, ageModel);
        Net genderNet = DnnInvoke.ReadNetFromCaffe(genderProto, genderModel);

        // Age & Gender labels
        string[] ageList = { "0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100" };
        string[] genderList = { "Male", "Female" };

        // Open the webcam
        var videoCapture = new VideoCapture(0);
        if (!videoCapture.IsOpened)
        {
            Console.WriteLine("Error: Webcam could not be opened!");
            return;
        }

        while (true)
        {
            var frame = videoCapture.QueryFrame();
            if (frame == null) continue;

            var grayImage = frame.ToImage<Gray, byte>();
            var faces = faceCascade.DetectMultiScale(grayImage, 1.1, 10, new Size(20, 20));

            foreach (var face in faces)
            {               
                Rectangle faceRect = new Rectangle(face.X, face.Y, face.Width, face.Height);
                faceRect.Intersect(new Rectangle(0, 0, frame.Width, frame.Height)); 

                if (faceRect.Width > 0 && faceRect.Height > 0) 
                {
                    using (Mat faceRegion = new Mat(frame, faceRect)) 
                    {
                        var faceImg = faceRegion.ToImage<Bgr, byte>().Resize(227, 227, Inter.Cubic);

                        var blob = DnnInvoke.BlobFromImage(faceImg, 1.0, new Size(227, 227),
                            new MCvScalar(78.4263377603, 87.7689143744, 114.895847746), false);

                        // Predict Gender
                        genderNet.SetInput(blob);
                        var genderPreds = genderNet.Forward();
                        int genderIndex = GetMaxIndex(genderPreds);
                        string gender = genderList[genderIndex];

                        // Predict Age
                        ageNet.SetInput(blob);
                        var agePreds = ageNet.Forward();
                        int ageIndex = GetMaxIndex(agePreds);
                        string age = ageList[ageIndex];

                        // Draw face rectangle
                        CvInvoke.Rectangle(frame, face, new Bgr(Color.Red).MCvScalar, 2);

                        // Draw age & gender prediction
                        string label = $"{gender}, {age}";
                        CvInvoke.PutText(frame, label, new Point(face.X, face.Y - 10),
                            FontFace.HersheySimplex, 0.8, new Bgr(Color.Yellow).MCvScalar, 2);
                    }
                }
            }

            CvInvoke.Imshow("Webcam - Face Detection with Age & Gender", frame);

            if (CvInvoke.WaitKey(1) == 27)
                break;
        }

        videoCapture.Dispose();
    }

    static int GetMaxIndex(Mat predictions)
    {
        float[] data = new float[predictions.Total.ToInt32()];
        predictions.CopyTo(data);
        return Array.IndexOf(data, data.Max());
    }
}
