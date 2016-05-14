using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Neural_Network_Project
{
    class Load_Data
    {
           #region  Variables 

        //private const string _training_dataset_path = "C:/Users/Muhammed Ramadan/Desktop/Neural_Network_Project/Neural_Network_Project/bin/Debug/DataSet/Training";
        //private const string _testing_dataset_path = "C:/Users/Muhammed Ramadan/Desktop/Neural_Network_Project/Neural_Network_Project/bin/Debug/DataSet/Testing";
        private const string _training_dataset_path = "C:/Users/Joe/Dropbox/Neural Nets Project/Project implementation/matlab/Data set/Training";
        private const string _testing_dataset_path = "C:/Users/Joe/Dropbox/Neural Nets Project/Project implementation/matlab/Data set/Testing";
        Dictionary<string, int> __objects = new Dictionary<string, int>();
        public List<KeyValuePair<int, float[]>> Training_data = new List<KeyValuePair<int, float[]>>();
        public List<KeyValuePair<int, float[]>> Testing_data = new List<KeyValuePair<int, float[]>>();
        SIFTDetector siftCPU = new SIFTDetector();
        VectorOfKeyPoint modelKeyPoints = new VectorOfKeyPoint();
        RBFN rbf;
        RadBFN rb = new RadBFN();
        SupportVectorMachine svm;
          #endregion

        public Load_Data()
        {
            __objects.Add(" Cat", 0);
            __objects.Add(" Laptop", 1);
            __objects.Add(" Apple", 2);
            __objects.Add(" Car", 3);
            __objects.Add(" Helicopter", 4);
            rbf = new RBFN();
            read_data();
        }

        private void read_data()
        {

            List<string> imgs_path = Directory.GetFiles(_training_dataset_path, "*.jpg", SearchOption.AllDirectories)
                .ToList();
            List<string> imgs_path_Test = Directory.GetFiles(_testing_dataset_path, "*.jpg", SearchOption.AllDirectories)
              .ToList();
            List<ImageFeature<float>[]> TrainD = new List<ImageFeature<float>[]>();
            List<int> desired = new List<int>();
            #region Loading Training
            foreach (var img_path in imgs_path)
            {
                string[] str = img_path.Split('-', '.');
                string objectType = str[1];
             
                Image<Gray, Byte> modelImage = new Image<Gray, byte>(img_path);
             
                MKeyPoint[] mKeyPoints = siftCPU.DetectKeyPoints(modelImage, null);
                modelKeyPoints.Push(mKeyPoints);
                ImageFeature<float>[] reulst = siftCPU.ComputeDescriptors(modelImage, null, mKeyPoints);
                Image<Bgr, Byte> image = Features2DToolbox.DrawKeypoints(modelImage, modelKeyPoints, new Bgr(Color.Red), Features2DToolbox.KeypointDrawType.DEFAULT);
                if (__objects.ContainsKey(objectType))
                {
                    int d = __objects[objectType];
                    setD(d, reulst);
                    desired.Add(d);
                }
                TrainD.Add(reulst);
               
            }
           // rbf.Train(Training_data, 5, 2, 0.0001);
          
            rb.Training(TrainD , desired,5, 2 , 0.0001);
            
            //SVM
            Matrix<float> trainData = new Matrix<float>(Training_data.Count, 128);
            Matrix<float> trainClasses = new Matrix<float>(Training_data.Count, 1);
            for (int i = 0; i < Training_data.Count; ++i)
            {
                KeyValuePair<int, float[]> item = Training_data[i];
                trainClasses[i, 0] = item.Key;
                for (int j = 0; j < 128; ++j)
                {
                    trainData[i, j] = item.Value[j];
                }

            }
            svm = new SupportVectorMachine(trainData, trainClasses);

            #endregion

            #region Loading Testing
            
          
            //foreach (var img_path in imgs_path_Test)
            //{
            //    string[] str = img_path.Split('-', '.');
            //    string objectType = str[1];
            //    Bitmap bm = new Bitmap(img_path);
            //    Image<Gray, Byte> modelImage = new Image<Gray, byte>(bm);
            //    SIFTDetector siftCPU = new SIFTDetector();
            //    VectorOfKeyPoint modelKeyPoints = new VectorOfKeyPoint();
            //    MKeyPoint[] mKeyPoints = siftCPU.DetectKeyPoints(modelImage, null);
            //    modelKeyPoints.Push(mKeyPoints);
            //    ImageFeature<float>[] reulst = siftCPU.ComputeDescriptors(modelImage, null, mKeyPoints);
            //    Image<Bgr, Byte> image = Features2DToolbox.DrawKeypoints(modelImage, modelKeyPoints, new Bgr(Color.Red), Features2DToolbox.KeypointDrawType.DEFAULT);

            //    string[] temp = objectType.Split();
            //    if (__objects.ContainsKey(objectType))
            //    {
            //        int d = __objects[objectType];
            //        steTest(d, reulst);
            //    }

            //}
          
        

            #endregion





        }
        public void Test()
        {
            Testing_data.Clear();
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.ShowDialog();
            string[] str = ofd.FileName.Split('-', '.');
            string objectType = str[1];
            str = objectType.Split();
            Image<Gray, Byte> modelImage = new Image<Gray, byte>(ofd.FileName);
            MKeyPoint[] mKeyPoints = siftCPU.DetectKeyPoints(modelImage, null);
            modelKeyPoints.Push(mKeyPoints);
            ImageFeature<float>[] reulst = siftCPU.ComputeDescriptors(modelImage, null, mKeyPoints);
            List<int> desired = new List<int>();
            foreach (var obj in str)
            {

                string t = " " + obj;
                if (__objects.ContainsKey(t))
                {
                    int d = __objects[t];
                    desired.Add(d);
                    //steTest(d, reulst);
                }
            }
          //  rbf.Test(Testing_data);
            rb.Testing(reulst,desired);
            TestSVM();
            
        }

        public void TestSVM()
        {
            Testing_data.Clear();
            List<string> imgs_path_Test = Directory.GetFiles(_testing_dataset_path, "*.jpg", SearchOption.AllDirectories)
              .ToList();
            List<Dictionary<int, int>> Confusion_matrix = new List<Dictionary<int, int>>();
            foreach (var img_path in imgs_path_Test)
            {
                string[] str = img_path.Split('-', '.');
                string objectType = str[1];
                str = objectType.Split();
                Image<Gray, Byte> modelImage = new Image<Gray, byte>(img_path);
                MKeyPoint[] mKeyPoints = siftCPU.DetectKeyPoints(modelImage, null);
                modelKeyPoints.Push(mKeyPoints);
                ImageFeature<float>[] reulst = siftCPU.ComputeDescriptors(modelImage, null, mKeyPoints);
                List<int> desired = new List<int>();
                foreach (var obj in str)
                {

                    string t = " " + obj;
                    if (__objects.ContainsKey(t))
                    {
                        int d = __objects[t];
                        desired.Add(d);
                        //steTest(d, reulst);
                    }
                }
                List<KeyValuePair<List<int>, float[]>> inSample = new List<KeyValuePair<List<int>, float[]>>(5);               
                foreach (ImageFeature<float> feature in reulst) 
                {
                    Matrix<float> sample = new Matrix<float>(1,128);
                    float[] _sample = new float[128];
                    for (int i = 0; i < 128; ++i)
                    {
                        sample[0, i] = feature.Descriptor[i];
                        _sample[i] = sample[0,i];
                    }
                    
                    KeyValuePair<List<int>, float[]> pair = new KeyValuePair<List<int>, float[]>(desired, _sample);
                    inSample.Add(pair);

                    
                }

                int[] _out = svm.Testing(inSample); 
                
            }
        }


        #region Helper Methods
        private void setD(int d, ImageFeature<float>[] item)
        {
            for (int i = 0; i < item.Count(); i++)
            {
                Training_data.Add(new KeyValuePair<int, float[]>(d, item[i].Descriptor));
            }
        }
        private void steTest(int d, ImageFeature<float>[] item)
        {
            for (int i = 0; i < item.Count(); i++)
            {
                Testing_data.Add(new KeyValuePair<int, float[]>(d, item[i].Descriptor));
            }
        }
        #endregion



    }
}
