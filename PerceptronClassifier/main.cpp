#include <stdio.h>
#include <time.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stack>
#include <list>
#include <vector>
#include <queue>
#include <chrono>
using namespace std;
class PerceptionClassifier
{
	private:
	vector<double> w;
	vector<vector<double> > i;
	vector<double> yin;
	vector<int> y;
	vector<int> testR;
	double b;
	double a;
	int ep;
	int datasize=0;
	int numchange=0;
	int numlooked;
	int lastepoch;
	bool fileerr=false;
	vector<double> out;
	string dataset;
	public:
	PerceptionClassifier(){
		w.assign(4,0.0);
		b=0;
		
	}
	/*vector<double> getI(int p){
		switch (p){
			case 0:
				return i0;
				break;
			case 1:
				return i1;
				break;
			case 2:
				return i2;
				break;
			case 3:
				return i3;
				break;
			default:
				break;
		}
		
	}*/
	void Train(bool info)
	{
		using namespace std::chrono;
		auto start = steady_clock::now();
		numlooked=(ep-1)*datasize;
		bool noChange=false;
		for (int p=0;p<=ep-1;p++)
		{
			lastepoch=0;
			if (!noChange)
			{
			noChange=true;
			if (info){cout<<"\nEpoch "<<p;}
		double sum=0;
		for (int j=0;j<datasize*0.8;j++){
		for (int d=0;d<=3;d++)
		{
			
			sum=sum+(w[d]*i[j][d]);
		}
			
			
			yin.push_back(sum+b);
			if ((sum+b)>0){
				y.push_back(1);
			}
			else 
			{
				y.push_back(-1);
			}
			sum=0;
			
			if(y[j]!=out[j])
				{
					noChange=false;
					numchange++;
					lastepoch++;
					for (int g=0;g<=3;g++)
					{
						
						w[g]=w[g]+a*out[j]*i[j][g];
						b=b+a*out[j];
						
					}
					
				} 	
		}
		y.erase(y.begin(),y.end());
		}
		
		else{cout<<"\nTraining converged on epoch "<<p-1<<"\n"; numlooked=(p-1)*(datasize);break;}
		if (info){cout<<"\n0: "<<w[0]<<" 1: "<<w[1]<<" 2: "<<w[2]<<" 3: "<<w[3]<<" b: "<<b<<"\n";}
		if (p==ep-1){cout<<"\nTraining did not converge over "<<ep<<" epochs";}
		}
		
		
			auto end = steady_clock::now();
			auto dif=end-start;
			double dur = duration<double, milli>(dif).count();	
			cout<<"\nTraining time: "<<dur<<"ms";
	}
		
	void Test()
	{
		int cor=0;
		int incor=0;
		double result=0;
		
		for (int j=datasize*0.8;j<datasize;j++){
		for (int d=0;d<=3;d++)
		{
			
			result=result+(w[d]*i[j][d]);
			
		}
		
		if ((result+b)>0){
				testR.push_back(1);
			}
			else 
			{
				testR.push_back(-1);
			}
			result=0;
		
		if (testR[j-datasize*0.8]==out[j])
		{
			cor++;
			
		}
		else
		{
			incor++;
			
		}
		}
		cout<<"Threshold: 0\n";
		cout<<"Training Accuracy (percentage of input/output pairs that did not change weights or bias over entire training): "<<double(double(numlooked-numchange)/double(numlooked)*100)<<"%\n";
		cout<<"Training Accuracy (percentage of input/output pairs that did not change weights or bias in the last epoch): "<<double(double(datasize-lastepoch)/double(datasize)*100)<<"%\n";
		cout<<"Testing Accuracy: "<<double(cor)/double(cor+incor)*100.0<<"%\n";
	}
	

	void unread()
	{
		i.erase(i.begin(),i.end());
		w.assign(4,0.0);
		yin.erase(yin.begin(),yin.end());
		y.erase(y.begin(),y.end());
		b=0;
		testR.erase(testR.begin(),testR.end());
		numchange=0;
		numlooked=0;
		datasize=0;
		lastepoch=0;
		out.erase(out.begin(),out.end());
	
	}
	bool readG(){return !fileerr;}
	void read(string name)
	{
		string line;
		string token;
		dataset="B";
		string diff="Iris-versicolor";
		int count=0;
		if (name=="datasetA.txt")
		{
			dataset="A";
			diff="Iris-setosa";
		}
		ifstream data (name.c_str());
		
	
		vector<double> pusher;
		
		if (data.is_open())
		{fileerr=false;
		getline(data,line)	;
		stringstream we(line);
		we>>a>>ep;
		cout<<a<<ep;
			
		while (getline (data,line))
		{	
			stringstream ss(line);
			while (getline(ss,token,','))
			{
				if (count==4)
				{
					if (token==diff){
					out.push_back(1.0);
					}
					else{
					out.push_back(-1.0);}
					count=0;
				}	
				
				else{
					
				pusher.push_back(atof(token.c_str()));
				count++;
				}
			}
			i.push_back(pusher);
			pusher.erase(pusher.begin(),pusher.end());
			datasize++;
		}
		}
		
		else{ cout<<"Can't Open File...\n";fileerr=true;}
		data.close();
	}	
};

int main(int argc, char **argv)
{
	PerceptionClassifier* p=new PerceptionClassifier;
	bool go=true;
	string fn;
	string op;
	while (go)
	{
		cout<<"--------Perceptron Classifier--------\nEnter Filename\n";
		cin>>fn;
		p->read(fn);
		if (p->readG())
		{
			cout<<"Read "<<fn<<" sucessfully...\n1) Train displaying weights and biases\n2) Train without display\n ";
			cin>>op;
			
			
			if (op=="1"){p->Train(true);} else {p->Train(false);}
			
			cout<<"\n1) Test Clasifier\n2) Change file\n";
			cin>>op;
			if (op=="1"){cout<<"Testing complete...\n---Results---\n";p->Test();} else {p->unread();}
			p->unread();
			cout<<"1) Change File\n2) Quit";
			cin>>op;
			if (op=="2"){go=false;}

		}
		
	
	}
	

	
	return 0;
}
