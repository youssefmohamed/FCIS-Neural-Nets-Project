using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network_Project
{
    class Perceptron
    {
        private double[] Weights;

        public double Bais;
        private double OutputY, OutputV, e;
        static Random Rand = new Random();
        public Perceptron(int num)
        {
           
            Weights = new double[num];

            for (int i = 0; i < num; i++)
            {
               // Weights[i] = 0;
                Weights[i] = Rand.NextDouble();
            }
        }
        public double get_outY()
        {
            return this.OutputY;
        }
        public double get_outV()
        {
            return this.OutputV;
        }
        public void set_out(double _outY, double _outV)
        {
            this.OutputY = _outY;
            this.OutputV = _outV;
        }
        public List<double> get_weights() // Temp for just now :))
        {
            List<double> temp = new List<double>();
            for (int i = 0; i < Weights.Count(); i++)
            {
                temp.Add(Weights[i]);
            }
            return temp;
        }
        public void set_weights(List<double> weight)
        {
            for (int i = 0; i < weight.Count(); i++)
            {
                this.Weights[i] = weight[i];
            }

        }


        public void set_error(double e)
        {
            this.e = e;
        }
        public double get_error()
        {
            return this.e;
        }

    }
}



