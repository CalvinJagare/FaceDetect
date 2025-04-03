using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System;
using System.Drawing;
using System.IO;

class Program
{
    static void Main(string[] args)
    {
        // Load the cascade classifier for face detection
        string faceCascadeFile = @"C:\Users\calda\source\repos\FaceDetectTest\FaceDetectTest\Resources\haarcascade_frontalface_default.xml";


        // Check if the file exists
        if (!File.Exists(faceCascadeFile))
        {
            Console.WriteLine($"Error: The file '{faceCascadeFile}' does not exist.");
            return;
        }

        // Initialize the cascade classifier
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

        // Open the webcam (use 0 for default webcam)
        var videoCapture = new VideoCapture(0);

        // Check if the webcam is opened successfully
        if (!videoCapture.IsOpened)
        {
            Console.WriteLine("Error: Webcam could not be opened!");
            return;
        }

        // Create a window to display the webcam feed using CvInvoke.Imshow
        while (true)
        {
            // Capture a frame from the webcam
            var frame = videoCapture.QueryFrame();

            // Convert the frame to grayscale using ToImage<Gray, byte>()
            var grayImage = frame.ToImage<Gray, byte>();

            // Detect faces in the frame
            var faces = faceCascade.DetectMultiScale(grayImage, 1.1, 10, new Size(20, 20));

            // Draw rectangles around detected faces
            foreach (var face in faces)
            {
                // You can use CvInvoke.Rectangle to draw the rectangle instead of Draw if there is an issue
                CvInvoke.Rectangle(frame, face, new Bgr(Color.Red).MCvScalar, 2);
            }

            // Display the frame with detected faces
            CvInvoke.Imshow("Webcam - Face Detection", frame);

            // Break the loop if the user presses a key
            if (CvInvoke.WaitKey(1) >= 0)
            {
                break;
            }
        }

        // Release the webcam and close the window
        videoCapture.Dispose();
    }
}

