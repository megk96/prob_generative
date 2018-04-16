#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <math.h>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Dense>
using Eigen::MatrixXd;
using namespace std;
#define N 4



void getCofactor(float A[N][N], float temp[N][N], int p, int q, int n)
{
    int i = 0, j = 0;

    																	// Looping for each element of the matrix
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            															//  Copying into temporary matrix only those element
            															//  which are not in given row and column
            if (row != p && col != q)
            {
                temp[i][j++] = A[row][col];

                														// Row is filled, so increase row index and
                														// reset col index
                if (j == n - 1)
                {
                    j = 0;
                    i++;
                }
            }
        }
    }
}


float determinant(float A[N][N], int n)
{
    int D = 0; // Initialize result

    																	//  Base case : if matrix contains single element
    if (n == 1)
        return A[0][0];

    float temp[N][N]; 													// To store cofactors

    int sign = 1;  

     																	// Iterate for each element of first row
    for (int f = 0; f < n; f++)
    {
        																// Getting Cofactor of A[0][f]
        getCofactor(A, temp, 0, f, n);
        D += sign * A[0][f] * determinant(temp, n - 1);

       			 														// terms are to be added with alternate sign
        sign = -sign;
    }

    return D;
}

																		// Function to get adjoint of A[N][N] in adj[N][N].
void adjoint(float A[N][N],float adj[N][N])
{
    if (N == 1)
    {
        adj[0][0] = 1;
        return;
    }

    																	// temp is used to store cofactors of A[][]
    float sign = 1, temp[N][N];

    for (int i=0; i<N; i++)
    {
        for (int j=0; j<N; j++)
        {
            															// Get cofactor of A[i][j]
            getCofactor(A, temp, i, j, N);

           																// sign of adj[j][i] positive if sum of row
            															// and column indexes is even.
            sign = ((i+j)%2==0)? 1: -1;

            															// Interchanging rows and columns to get the
           																// transpose of the cofactor matrix
            adj[j][i] = (sign)*(determinant(temp, N-1));
        }
    }
}


bool inverse(float A[N][N], float inverse[N][N])
{
    																	// Find determinant of A[][]
    int det = determinant(A, N);
    if (det == 0)
    {
        cout << "Singular matrix, can't find its inverse";
        return false;
    }

    																	// Find adjoint
    float adj[N][N];
    adjoint(A, adj);

    																	// Find Inverse using formula "inverse(A) = adj(A)/det(A)"
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
            inverse[i][j] = adj[i][j]/float(det);

    return true;
}
template<class T>
void display(T A[N][N])
{
    for (int i=0; i<N; i++)
    {
        for (int j=0; j<N; j++)
            cout << A[i][j] << " ";
        cout << endl;
    }
}

																		//standard code to read every line in a CSV file
class CSVRow
{
    public:
        string const& operator[](size_t index) const
        {
            return m_data[index];
        }
        size_t size() const
        {
            return m_data.size();
        }
        void readNextRow(istream& str)
        {
            string         line;
            getline(str, line);

            stringstream   lineStream(line);
            string         cell;

            m_data.clear();
            while(getline(lineStream, cell, ','))
            {
                m_data.push_back(cell);
            }
          
            if (!lineStream && cell.empty())
            {

                m_data.push_back("");
            }
        }
    private:
        vector<string>    m_data;
};

istream& operator>>(istream& str, CSVRow& data)
{
    data.readNextRow(str);
    return str;
}   
int main()
{
    ifstream       file("train.txt");
	float m1[4] = {0.0},m2[4] = {0.0};
	float s1[4] = {0.0},s2[4] = {0.0};
    CSVRow row;
	float N1=0, N2;
	int x;
    while(file >> row)
    {
		stringstream num(row[4]);
		num >> x;
        if(x==1)															//find number of positive examples
			N1++;
		for(int i=0;i<4;i++)
			{
				float y;
				stringstream data(row[i]);
				data >> y;
				m1[i] += x*y;
				m2[i] += (1-x)*y;

			}	
	
    }
	N2 = 960-N1; 															//number of negative examples
	for(int i=0;i<4;i++)
	{
		m1[i] /=N1;
																			//mean of positive and negative examples
		m2[i] /=N2;
	}
	

	file.clear();
	file.seekg(0);
	float S1[4][4] = {0.0}, S2[4][4]= {0.0};
	float S[4][4];
	while(file >> row)
    {
		stringstream num(row[4]);
		num >> x;
		for(int i=0;i<4;i++)
			{
				float y;
				stringstream data(row[i]);
				data >> y;													//covariance of each class
				s1[i] = x*(y-m1[i]);
				s2[i] = (1-x)*(y-m2[i]);

			}
		for(int i=0;i<4;i++)
			for(int j = 0;j<4;j++)
				{
					S1[i][j] += s1[i]*s1[j];
					S2[i][j] += s2[i]*s2[j];
				}
			
	
    }
	for(int i=0;i<4;i++)													//covariance matrix of both the classes combined
			{for(int j = 0;j<4;j++)
				{	
					S[i][j] = (S1[i][j]+S2[i][j])/960;
					
			
				}
				
				
			}
	MatrixXd inve(4,4);														//inverse of the matrix
	
	MatrixXd sigma(4,4);
	
	for(int i=0;i<4;i++)
		for(int j = 0;j<4;j++)
			sigma(i,j) = S[i][j];
	
	inve=sigma.inverse();
 
    
    float inv[4][4];
    
    for(int i=0;i<4;i++)
		for(int j = 0;j<4;j++)
			inv[i][j] = inve(i,j);
  
		
  float m[4] = {0.0};
	for(int i=0;i<4;i++)	
		m[i] = m1[i]-m2[i]; 
	float w[4]={0.0};
	for(int i=0;i<4;i++)
		for(int j=0;j<4;j++)
			w[i]+=inv[i][j]*m[j];											//calculate w vector
	cout << "w: ";
	for(int i=0;i<4;i++)
		cout << w[i] << " "; 
	cout << endl;		
	
	float w1[4]={0.0};
	for(int i=0;i<4;i++)
		for(int j=0;j<4;j++)
			w1[i]+=inv[i][j]*m1[j];									
	float z =0,z1=0;
	for(int i=0;i<4;i++)
		z += w1[i]*m1[i];
	
	float w2[4]={0.0};
	for(int i=0;i<4;i++)
		for(int j=0;j<4;j++)
			w2[i]+=inv[i][j]*m2[j];
	for(int i=0;i<4;i++)
		z1 += w2[i]*m2[i];
	float w0 = -0.5*z + 0.5*z1 +log(N1/N2);									//calculate w0
	
	cout << "w0:  " << w0 <<endl << endl;
	
	ifstream       test("test.txt");
	
	float truepos=0;
	float trueneg=0;
	float falsepos=0;
	float falseneg=0;

	while(test >> row)
    {
    	int correct,pred;
   
		stringstream num(row[4]);
		num >> correct;
		float sig=0;
		for(int i=0;i<4;i++)
			{
				float y;
				stringstream data(row[i]);
				data >> y;
				sig += w[i]*y;

			}	
			sig += w0;
			sig = 1/(1+exp(-sig));
			if(sig>0.5) 
			{
				pred = 1;
				if(pred==correct) truepos++;
				else falsepos++;
			}
			else
			{
				pred  = 0;
				if(pred==correct) trueneg++;
				else falseneg++;
			}
			
			 
	
    }

	cout<<endl<<"CONFUSION MATRIX"<<endl;
    cout<<"\t\t"<<"Predicted 0\tPredicted 1"<<endl;
    cout<<"Actual 0\t"<<trueneg<<"\t\t"<<falsepos<<"\t\t|\t"<<trueneg+falsepos<<endl;
    cout<<"Actual 1\t"<<falseneg<<"\t\t"<<truepos<<"\t\t|\t"<<falseneg+truepos<<endl;
    cout<<"               _________________________________|___________"<<endl;
    cout<<"\t\t"<<trueneg+falseneg<<"\t\t"<<falsepos+truepos<<"\t\t|\t"<<trueneg+falsepos+falseneg+truepos<<endl;

	cout << endl;

	

	float accuracy=(truepos+trueneg)/(truepos+trueneg+falsepos+falseneg);
	float precisionpos=(truepos)/(truepos+falsepos);
	float precisionneg=(trueneg)/(trueneg+falseneg);
	float recallpos=(truepos)/(truepos+falseneg);
	float recallneg=(trueneg)/(trueneg+falsepos);

	cout<<"Accuracy on testing data is: "<<accuracy*100<<"%\n";
	cout<<"Precision pos on testing data is: "<<precisionpos*100<<"%\n";
	cout<<"Recall pos on testing data is: "<<recallpos*100<<"%\n";
	cout<<"Precision neg on testing  data is: "<<precisionneg*100<<"%\n";
	cout<<"Recall neg on testing  data is: "<<recallneg*100<<"%\n";

	
	cout << endl << endl;

	cout<<"F-Measure\t"<<2*precisionneg*recallneg/(precisionneg+recallneg)*100<<" %\t"<<2*precisionpos*recallpos/(precisionpos+recallpos)*100<<" %"<<endl<<endl;	
	ifstream       train("train.txt");
	
	truepos=0;
	trueneg=0;
	falsepos=0;
	falseneg=0;

	while(train >> row)
    {
    	int correct,pred;
   
		stringstream num(row[4]);
		num >> correct;
		float sig=0;
		for(int i=0;i<4;i++)
			{
				float y;
				stringstream data(row[i]);
				data >> y;
				sig += w[i]*y;

			}	
			sig += w0;
			sig = 1/(1+exp(-sig));
			if(sig>0.5) 
			{
				pred = 1;
				if(pred==correct) truepos++;
				else falsepos++;
			}
			else
			{
				pred  = 0;
				if(pred==correct) trueneg++;
				else falseneg++;
			}
			
			 
	
    }
	

	accuracy=(truepos+trueneg)/(truepos+trueneg+falsepos+falseneg);
	precisionpos=(truepos)/(truepos+falsepos);
	precisionneg=(trueneg)/(trueneg+falseneg);
	recallpos=(truepos)/(truepos+falseneg);
	recallneg=(trueneg)/(trueneg+falsepos);

	cout<<"Accuracy on training data is: "<<accuracy*100<<"%\n";
	cout<<"Precision pos on training data is: "<<precisionpos*100<<"%\n";
	cout<<"Recall pos on training data is: "<<recallpos*100<<"%\n";
	cout<<"Precision neg on training  data is: "<<precisionneg*100<<"%\n";
	cout<<"Recall neg on training  data is: "<<recallneg*100<<"%\n";




}
