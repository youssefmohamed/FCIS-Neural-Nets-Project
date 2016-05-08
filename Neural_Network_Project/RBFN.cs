using Neural_Network_Project;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Neural_Network_Project
{
    class RBFN
    {
        #region Variables
        List<sample> Training_data = new List<sample>();
        List<sample> old_centers = new List<sample>();
        List<cluster> Cluster = new List<cluster>();
        List<double> segma = new List<double>();
        List<Perceptron> OutPut = new List<Perceptron>(5);
        string[] objects = new string[5];
        List<sample> Test_data = new List<sample>();
        #endregion



        // Cluster Structure 
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


       public RBFN()
        {
            objects[0] = " Cat";
            objects[1] = " Laptop";
            objects[2] = " Apple";
            objects[3] = " Car";
            objects[4] = " Helicopter";
        }

        #region De7tk
        private void kMeans(int k)
        {

           
            #region Create Output layer
            for (int i = 0; i < 5; i++)  // Construct the output Layer
            {
                Perceptron per = new Perceptron(k);
                OutPut.Add(per);
            }
            #endregion



            #region Get_Cntres
            // Get Random Centers For Clusters 
            Random rd = new Random();
            for (int i = 0; i < k; i++)
            {
                cluster temp = new cluster();
                temp.center = Training_data[rd.Next(0, Training_data.Count)];

                Cluster.Add((temp));

                KeyValuePair<int, float[]> ls = new KeyValuePair<int, float[]>();
                sample s = new sample();
                s.copysample(temp.center);
                old_centers.Add(s);

            }
            #endregion

            //    old_centers[0].Value[0] = Training_data[rd.Next(0,Training_data.Count)].Value[0];
            old_centers[0].features[0] = (float)-0.0002;

            while (chek())
            {

                List<double> _out = new List<double>();
                #region updateing_old_centers
                for (int i = 0; i < k; i++)
                {

                    Cluster[i].cluster_samples.Clear();

                    old_centers[i].copysample(Cluster[i].center);
                }
                #endregion

                #region Clustring
                //Clustering   // Here's a trap
                for (int i = 0; i < Training_data.Count; i++)   // outer one
                {
                    //for (int z = 0; z < Training_data[i].Count; z++)
                    {

                        _out.Clear();
                        for (int j = 0; j < k; j++)
                        {
                            _out.Add(distance(Training_data[i].features, Cluster[j].center.features));
                        }
                        Cluster[get_min(_out)].cluster_samples.Add(Training_data[i]);
                    }

                }
                #endregion


                #region   get new centers
                for (int c = 0; c < k; c++)
                {

                    double[] x = new double[128];
                    for (int i = 0; i < Cluster[c].cluster_samples.Count; i++)
                    {

                        for (int j = 0; j < 128; j++)
                        {
                            x[j] += Convert.ToDouble(Cluster[c].cluster_samples[i].features[j]);
                        }
                    }
                    if (Cluster[c].cluster_samples.Count != 0)
                    {

                        
                        for (int sd = 0; sd < 128; sd++)
                        {
                            x[sd] /= Cluster[c].cluster_samples.Count;
                        }
                        //  Cluster[c].center.;
                        for (int sd = 0; sd < 128; sd++)
                        {

                            Cluster[c].center.features[sd] = (float)(x[sd]);


                        }

                    }
                }


                #endregion


            }

        }
        public void Train(List<KeyValuePair<int, float[]>> TData, int k, int epochs, double eta)
        {
            //Training_data = TData;
            set_IT(TData);
            kMeans(k);
            List<double> segma = get_Seg();
            for (int epoch = 0; epoch < epochs; epoch++)
            {

                for (int i = 0; i < Training_data.Count; i++)
                {
                    List<double> _input = new List<double>();
                    for (int j = 0; j < k; j++)
                    {
                        _input.Add(distance(Training_data[i].features, Cluster[j].center.features));
                    }

                    for (int O = 0; O < OutPut.Count; O++)
                    {
                        List<double> temp = OutPut[O].get_weights();
                        double sumv = 0;
                        // List<double> segma = get_Seg();
                        for (int z = 0; z < temp.Count; z++)
                        {
                            sumv += Math.Exp(-1 * (_input[z] * _input[z]) / (2 * segma[z] * segma[z])) * temp[z];
                        }
                        OutPut[O].set_out(sumv, sumv);

                        for (int z = 0; z < temp.Count; z++)  // updating weights
                        {
                            temp[z] += eta * (Training_data[i].d - sumv) * _input[z];
                        }
                        OutPut[O].set_weights(temp);
                    }


                }
            }



        }
        public void Test(List<KeyValuePair<int, float[]>> TData)
        {
            
            set_ITest(TData);
            List<double> _outp = new List<double>();
            int[] res = new int[5];
            int _True = 0, _False = 0;
            List<double> segma = get_Seg();

            for (int i = 0; i < Test_data.Count; i++)
            {
                _outp.Clear();
                List<double> _input = new List<double>();
                for (int j = 0; j < Cluster.Count; j++)
                {
                    _input.Add(distance(Test_data[i].features, Cluster[j].center.features));
                }

                for (int O = 0; O < OutPut.Count; O++)
                {
                    List<double> temp = OutPut[O].get_weights();
                    double sumv = 0;
                    for (int z = 0; z < temp.Count; z++)
                    {
                        //sumv += (_input[z] / (2 * segma[z])) * temp[z];     
                        sumv += Math.Exp(-1 * (_input[z] * _input[z]) / (2 * segma[z] * segma[z])) * temp[z];
                    }
                    _outp.Add(sumv);


                }



                List<int> __outp = get_out(_outp);

                for (int cc = 0; cc < OutPut.Count; cc++)
                {
                    res[cc] += __outp[cc];
                }

                List<string> FinalOut = new List<string>();



            }
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
        }

        #endregion



        #region Private Methods

        private bool chek()
        {
            int cntr = 0;
            for (int i = 0; i < Cluster.Count; i++)
            {

                int c = 0;
                for (int k = 0; k < 128; k++)
                {
                    if (Cluster[i].center.features[k] == old_centers[i].features[k]) c++;
                }

                if (c == 128) cntr++;

            }
            if (cntr == Cluster.Count) return false;
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

        private List<double> get_Seg()
        {
            List<double> seg = new List<double>();
            for (int i = 0; i < Cluster.Count; i++)
            {
                double meu = 0;
                List<double> dis = new List<double>();
                for (int j = 0; j < Cluster[i].cluster_samples.Count; j++)
                {
                    dis.Add(distance(Cluster[i].cluster_samples[j].features, Cluster[i].center.features));
                    dis[j] *= dis[j];
                    meu += dis[j];
                }
                meu /= Cluster[i].cluster_samples.Count;

                seg.Add(meu);
            }
            return seg;
        }

        private int get_min(List<double> ls)
        {
            double s = ls[0];
            int res = 0;
            for (int i = 1; i < ls.Count; i++)
            {
                if (ls[i] < s)
                {
                    res = i;
                    s = ls[i];
                }
            }
            return res;
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
        private void set_IT(List<KeyValuePair<int, float[]>> TData)
        {
            for (int i = 0; i < TData.Count; i++)
            {
                sample s = new sample();
                s.features = TData[i].Value;
                s.d = TData[i].Key;
                Training_data.Add(s);
            }
        }
        private void set_ITest(List<KeyValuePair<int, float[]>> TData)
        {
            for (int i = 0; i < TData.Count; i++)
            {
                sample s = new sample();
                s.features = TData[i].Value;
                s.d = TData[i].Key;
                Test_data.Add(s);
            }
        }

        #endregion



    }
}
