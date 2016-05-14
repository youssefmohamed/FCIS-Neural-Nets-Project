using Emgu.CV;
using Emgu.CV.ML;
using Emgu.CV.ML.MlEnum;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV.ML.Structure;

namespace Neural_Network_Project
{
    class SupportVectorMachine
    {
        Matrix<float> sample = new Matrix<float>(1, 128);
        SVM model1 = new SVM();
        SVM model2 = new SVM();
        public   SupportVectorMachine(Matrix<float> trainData, Matrix<float> trainClasses)
        {

            {
                
                SVMParams p = new SVMParams();
                p.KernelType = Emgu.CV.ML.MlEnum.SVM_KERNEL_TYPE.LINEAR;
                p.SVMType = Emgu.CV.ML.MlEnum.SVM_TYPE.C_SVC;
                p.C = 1;
                p.TermCrit = new MCvTermCriteria(100, 0.00001);

                bool trained1 = model1.Train(trainData, trainClasses, null, null, p);
                //bool trained2 = model2.TrainAuto(trainData, trainClasses, null, null, p.MCvSVMParams, 5);
               
            }
            
        }

      public  int[] Testing(List<KeyValuePair<List<int> , float[]>> inSample)
        {
            int[] _out = {0,0,0,0,0};
            for (int i = 0; i < inSample.Count; i++)  // to number of keypoints 
             {
                for (int j = 0; j < 128; j++)
                {
                    sample.Data[0, j] = inSample[i].Value[j];  // = descriptor[j];
                    
                }
                
                float response1 = model1.Predict(sample); // represent class index  
                _out[Convert.ToInt32(response1)] += 1;
            }
            
          


            return _out;
        }

    }
}
