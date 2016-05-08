using Emgu.CV.Features2D;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Neural_Network_Project
{
    class RadBFN
    {



        #region Variables 
        List<sample> Training_data = new List<sample>();
        List<sample> Testing_data = new List<sample>();
        List<sample> old_centers = new List<sample>();
        List<cluster> Clusters = new List<cluster>();
        List<double> segma = new List<double>();
        List<Perceptron> OutPut = new List<Perceptron>(5);
        string[] objects = new string[5];
        #endregion

        public RadBFN()
        {
            objects[0] = " Cat";
            objects[1] = " Laptop";
            objects[2] = " Apple";
            objects[3] = " Car";
            objects[4] = " Helicopter";
        }

        #region Helper Structure 
        class cluster
        {
            public List<sample> cluster_samples;
            public sample center;
            public cluster()
            {
                cluster_samples = new List<sample>();
                center = new sample();
            }

        }

        class sample
        {

            public float[] features;
            public int d;
            public sample()
            {
                features = new float[128];
            }
            public void copysample(sample s)
            {
                this.d = s.d;
                s.features.CopyTo(this.features, 0);
            }
        }

        #endregion


        #region K-Means
        void K_Means(int K)
        {
            #region Create Output layer
            for (int i = 0; i < 5; i++)  // Construct the output Layer
            {
                Perceptron per = new Perceptron(K);
                OutPut.Add(per);
            }
            #endregion

            #region Get_Cntres
            // Get Random Centers For Clusters 
            Random rd = new Random();
            for (int i = 0; i < K; i++)
            {
                cluster temp = new cluster();
                temp.center = Training_data[rd.Next(0, Training_data.Count)];

                Clusters.Add((temp));

              
                sample s = new sample();
                s.copysample(temp.center);
                old_centers.Add(s);

            }
            #endregion


            old_centers[0].features[0] = (float)-0.0002;  // To change it for first time 

            while (check() )
            {
                #region updateing_old_centers
                for (int i = 0; i < K; i++)
                {

                    Clusters[i].cluster_samples.Clear();

                    old_centers[i].copysample(Clusters[i].center);
                }
                #endregion

                #region Clustering
                List<double> _out = new List<double>();
                for (int sampleindx = 0; sampleindx < Training_data.Count(); sampleindx ++ )
                {
                    _out.Clear();

                    for (int Clusterindx = 0; Clusterindx < Clusters.Count(); Clusterindx ++ )
                    {
                        _out.Add(distance(Training_data[sampleindx].features, Clusters[Clusterindx].center.features));
                    }

                    Clusters[get_min(_out)].cluster_samples.Add(Training_data[sampleindx]);
                }
                #endregion


                #region  Get New Centers 

                for (int clusterindx = 0; clusterindx < Clusters.Count(); clusterindx ++ )
                {
                    double[] x = new double[Clusters[clusterindx].center.features.Count()];
                    for (int sampleindx = 0 ; sampleindx < Clusters[clusterindx].cluster_samples.Count ; sampleindx++)
                    {
                    
                 
                    for (int featureindx = 0; featureindx < Clusters[clusterindx].center.features.Count(); featureindx++ )
                    {
                        x[featureindx] += Clusters[clusterindx].cluster_samples[sampleindx].features[featureindx];
                    }

                    }

                    for (int featureindx = 0; featureindx < Clusters[clusterindx].center.features.Count(); featureindx++)
                    {
                       
                        x[featureindx] /= Clusters[clusterindx].cluster_samples.Count();
                        Clusters[clusterindx].center.features[featureindx] = (float) x[featureindx]; // update new centers

                    }




                }

                #endregion
            }


        }

        #endregion

        #region Public Methods
        public void Training(List<ImageFeature<float>[]> TrainingData , List<int> desired , int k , int epochs , double eta)
        {
            DataProcessing(TrainingData, desired);
            K_Means(k);
            segma = get_Seg();
            for (int epoch = 0; epoch < epochs; epoch ++ )
            {
                for (int sampleindex = 0; sampleindex < Training_data.Count; sampleindex++)
                {

                    #region Getting Hidden layer Output
                    List<double> HidOut = new List<double>();

                    for (int clusterindx = 0; clusterindx < Clusters.Count(); clusterindx ++ )
                    {
                        double seg = segma[clusterindx];
                        HidOut.Add(distance(Training_data[sampleindex].features, Clusters[clusterindx].center.features));
                        HidOut[clusterindx] = -1 * Math.Pow(HidOut[clusterindx], 2);
                         seg = 2*Math.Pow(seg, 2);
                        HidOut[clusterindx] = Math.Exp(HidOut[clusterindx] / seg);
                    }

                    #endregion

                    #region Calculate Output Layer 
                    for (int Outindx = 0; Outindx < OutPut.Count(); Outindx ++ )
                    {
                       List<double> weights = OutPut[Outindx].get_weights();
                       double sumv = 0;
                       for (int weightindx = 0; weightindx < weights.Count(); weightindx++ )
                       {
                           sumv += (weights[weightindx] * HidOut[weightindx]);
                       }
                       OutPut[Outindx].set_out(sumv, sumv);

                        #region updating weights 
                       for (int weightindx = 0; weightindx < weights.Count(); weightindx++)
                       {
                           weights[weightindx] += eta * (Training_data[sampleindex].d - sumv) * HidOut[weightindx] ;
                       }
                       OutPut[Outindx].set_weights(weights);
                        #endregion


                    }
                    #endregion



                }
            }

        }

        public void Testing(  ImageFeature<float>[] TestImage , List<int> desired)
        {
            Testing_data.Clear();
            int[] res = new int[OutPut.Count()];
            TestDataProcessing(TestImage);
            for (int sampleindex = 0; sampleindex < Testing_data.Count; sampleindex++)
            {

                #region Getting Hidden layer Output
                List<double> HidOut = new List<double>();

                for (int clusterindx = 0; clusterindx < Clusters.Count(); clusterindx++)
                {
                    double seg = segma[clusterindx];
                    HidOut.Add(distance(Testing_data[sampleindex].features, Clusters[clusterindx].center.features));
                    HidOut[clusterindx] = -1 * Math.Pow(HidOut[clusterindx], 2);
                    seg= 2 *Math.Pow(seg, 2);
                    HidOut[clusterindx] = Math.Exp(HidOut[clusterindx] / seg);
                }

                #endregion


                #region Calculate Output Layer
                List<double> _outp = new List<double>();
              

                for (int Outindx = 0; Outindx < OutPut.Count(); Outindx++)
                {
                    List<double> weights = OutPut[Outindx].get_weights();

                    double sumv = 0;
                    for (int weightindx = 0; weightindx < weights.Count(); weightindx++)
                    {
                        sumv += (weights[weightindx] * HidOut[weightindx]);
                    }
                    _outp.Add(sumv);
                   

                }
                List<int> __outp = get_out(_outp);

                for (int cc = 0; cc < OutPut.Count; cc++)
                {
                    res[cc] += __outp[cc];
                }

                #endregion

            

            }
            #region Find Detected Objects

            string str = "";
            for (int ee = 0; ee < OutPut.Count; ee++)
            {
                if (res[ee] >= 3)
                {
                    // object of ee is exist in the image XD
                    str += objects[ee];
                    str += "\t";
                }
            }
            MessageBox.Show(str + " Is Exist\n");
            #endregion
        }

        #endregion

        #region Helper Methods
        void DataProcessing(List<ImageFeature<float>[]> TrainD , List<int> desired )
        {
            for (int imgeindx = 0; imgeindx < TrainD.Count();imgeindx++ )
            { 
                for (int descindx = 0 ; descindx < TrainD[imgeindx].Count();descindx++)
                {
                    sample temp = new sample();
                    for (int i = 0; i < TrainD[imgeindx][descindx].Descriptor.Count(); i++ )
                    {
                        temp.features[i] = TrainD[imgeindx][descindx].Descriptor[i];
                        temp.d = desired[imgeindx];
                    }
                    Training_data.Add(temp);
                }
            }
        }
        bool check()
        {
            int cntr = 0;
            for (int i = 0; i < Clusters.Count; i++)
            {

                int c = 0;
                for (int k = 0; k < 128; k++)
                {
                    if (Clusters[i].center.features[k] == old_centers[i].features[k]) c++;
                }

                if (c == 128) cntr++;

            }
            if (cntr == Clusters.Count) return false;
            return true;
           
        }
        private double distance(float[] s, float[] centre)
        {
            double res = 0.0;
            double diff;
            for (int i = 0; i < s.Count(); i++)
            {
                diff = ((s[i]) - (centre[i]));
                diff *= diff;
                res += diff;
            }
            res = Math.Sqrt(res);


            return res;
        }
         private int get_min(List<double> ls)
        {
            double _min = ls[0];
            int indx = 0;
            for (int i = 1; i < ls.Count(); i++ )
            {
                if (_min > ls[i])
                {
                    _min = ls[i];
                    indx = i;
                }
            }
            return indx;
        }
         private List<double> get_Seg()
         {
             List<double> seg = new List<double>();
             for (int i = 0; i < Clusters.Count; i++)
             {
                 double meu = 0;
                 List<double> dis = new List<double>();
                 for (int j = 0; j < Clusters[i].cluster_samples.Count; j++)
                 {
                     dis.Add(distance(Clusters[i].cluster_samples[j].features, Clusters[i].center.features));
                     dis[j] *= dis[j];
                     meu += dis[j];
                 }
                 meu /= Clusters[i].cluster_samples.Count;

                 seg.Add(meu);
             }
             return seg;
         }
         void TestDataProcessing(ImageFeature<float>[] TrainD)
         {
             for (int imgeindx = 0; imgeindx < TrainD.Count(); imgeindx++)
             {
                // for (int descindx = 0; descindx < TrainD[imgeindx].Count(); descindx++)
                 {
                     sample temp = new sample();
                     for (int i = 0; i < TrainD[imgeindx].Descriptor.Count(); i++)
                     {
                         temp.features[i] = TrainD[imgeindx].Descriptor[i];
                        
                     }
                     Testing_data.Add(temp);
                 }
             }
         }
         private List<int> get_out(List<double> temp)
         {
             List<int> back = new List<int>();
             back.Add(0);
             back.Add(0);
             back.Add(0);
             back.Add(0);
             back.Add(0);
             double mx = temp[0];
             int indx = 0;
             for (int i = 1; i < temp.Count; i++)
             {
                 if (mx < temp[i])
                 {
                     mx = temp[i];
                     indx = i;
                 }
             }
             back[indx] = 1;
             return back;
         }
        #endregion

    }
}
