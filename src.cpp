#include<opencv.hpp>
#include<iostream>
#include<highgui/highgui.hpp>
#include<ml/ml.hpp>
#include<string>
#include<fstream>
#include<vector>
#include<objdetect.hpp>
#include<stdlib.h>
#include<algorithm>
#include<io.h>

using namespace std;
using namespace cv;

vector<float>myDetector;	//Get your own detector
vector<string>FILENAME;	//The name of the positive sample
string Forward = "D:\\traindatabase\\pos\\";
string Back = ".jpg";
int SumPicture = 500;	//The number of positive sample

/*Train the svm module,and get your own detect unit*/
void Train_SVMmodel(const string& data_path,const string& save_path)
{
	int ImgWidght = 64;
	int ImgHeight = 128;
	vector<string> img_path;
	vector<int> img_catg;
	int nLine = 0;
	string buf;
	ifstream svm_data(data_path);
	unsigned long n;
	while (svm_data)
	{
		if (getline(svm_data, buf))
		{
			nLine++;
			if (nLine % 2 == 0)
			{
				img_catg.push_back(atoi(buf.c_str()));	//atoi tranfer the string to int form data，and marked（0，1）
			}
			else
			{
				img_path.push_back(buf);	//the path of img
			}
		}
	}
	svm_data.close();
	Mat data_mat, res_mat;
	int nImgNum = nLine / 2;
	data_mat = Mat::zeros(nImgNum, 3780, CV_32FC1);
	res_mat = Mat::zeros(nImgNum, 1, CV_32SC1);
	Mat src;
	Mat small;
	Mat trainImg = Mat(Size(ImgWidght, ImgHeight), 8, 3);
	for (string::size_type i = 0; i != img_path.size(); i++)
	{
		src = imread(img_path[i].c_str());
		if (src.empty())
		{
			cout << "can not load the image" << img_path[i] << endl;
			continue;
		}
		cout << "processing" << img_path[i].c_str() << endl;
		resize(src, small, Size(ImgWidght, ImgHeight));
		//cvtColor(small, small, COLOR_RGB2GRAY);
		HOGDescriptor *hog = new HOGDescriptor(Size(ImgWidght, ImgHeight), Size(16, 16), Size(8, 8), Size(8, 8), 9);
		vector<float>descriptors;
		hog->compute(small, descriptors, Size(8, 8));
		cout << "HOG dimision is" << descriptors.size() << endl;
		n = 0;
		for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
		{
			data_mat.at<float>(i, n) = *iter;
			n++;
		}
		res_mat.at<float>(i, 0) = img_catg[i];
		cout << "end processing " << img_path[i].c_str() << img_catg[i] << endl;
	}
	//cout << data_mat << endl;
	//cout << res_mat << endl;
	Ptr<ml::SVM>svm = ml::SVM::create();
	cout << "training" << endl;
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::LINEAR);
	svm->setDegree(10.0);
	svm->setGamma(8.0);
	svm->setCoef0(1.0);
	svm->setC(10.0);
	svm->setNu(0.5);
	svm->setP(0);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 3000, 1e-6));
	svm->train(data_mat, ml::ROW_SAMPLE, res_mat);
	cout << "End of training" << endl;
	svm->save(save_path);
	//+TermCriteria::EPS
}

/*get your own detect unit*/
void Detect_Unit(const string& filename)
{
	//HOGDescriptor hog(Size(32, 64), Size(4, 4), Size(8, 8), Size(8, 8), 9);
	int DescriptorDim;
	Ptr<ml::SVM>svm = ml::SVM::load(filename);
	DescriptorDim = svm->getVarCount();
	Mat supportVector = svm->getSupportVectors();
	int supportVectorNum = supportVector.rows;
	cout << "The number of support vectors is" << supportVectorNum << endl;
	vector<float>svm_alpha;
	vector<float>svm_svidx;
	float svm_rho;

	svm_rho = svm->getDecisionFunction(0, svm_alpha, svm_svidx);
	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);
	supportVectorMat = supportVector;
	//Copy alpha vector data to alphaMat and return alpha vector in the dicision function of SVM
	for (int i = 0; i < supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = svm_alpha[i];
	}
	//caluculate-（alphaMat*supportVectorMat），and put the output in the resultMat
	resultMat = -1 * alphaMat * supportVectorMat;
	vector<float>myDetector;
	//copy the data in  the resultMat to array myDetector;
	for (int i = 0; i < DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//最后添加偏移量rho,得到检测子Finally,add the offset,get your own detector.
	myDetector.push_back(svm_rho);
	HOGDescriptor myHOG;
	Size s1(16,16);
	Size s2(8, 8);
	myHOG.winSize = Size(64,128);
	myHOG.blockSize = s1;
	myHOG.blockStride = s2;
	myHOG.cellSize = s2;
	myHOG.nbins = 9;
	if (_access("D:\\svm\\svm\\HOGDetectorForOpenCv.txt",0)== -1)
	{
		ofstream fout("HOGDetectorForOpenCv.txt");
		for (unsigned int i = 0; i < myDetector.size(); i++)
		{
			fout << myDetector[i] << endl;
		}
	}
}

/*Load detector*/
/*if you have got detector,you could skip this step*/
void Load_Detect_Vector()
{
	/*Loaded when myDetector is empty*/
	ifstream finPos("D:\\svm\\svm\\HOGDetectorForOpenCv.txt");
	string buf;
	int nLine = 0;
	while (finPos)
	{
		if (getline(finPos, buf))
		{
			myDetector.push_back(atof(buf.c_str()));
			nLine++;
		}
	}
	finPos.close();
}

/*Init HOG eigen bector*/
/*void Init_HOG_Detector()
{
	Size s1(16, 16);
	Size s2(8, 8);
	myHOG.winSize = Size(64, 128);
	myHOG.blockSize = s1;
	myHOG.blockStride = s2;
	myHOG.cellSize = s2;
	myHOG.nbins = 9;
}*/

/*Finally,detect buttle*/
void Detect_Final()
{
	/*VideoCapture detect(0);
	if (!detect.isOpened())
	{
		cout << "ERROR" << endl;
		system("pause");
	}*/
	//Mat frame;
	//Mat src;D:\traindatabase\pos\0.jpg
	HOGDescriptor myHOG(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);	//Get your own HOG eigen vector
	myHOG.setSVMDetector(myDetector);
	VideoCapture detect(0);
	Mat inter;
	if (!detect.isOpened())
	{
		cout << "ERROR" << endl;
		system("pause");
	}
	while (true)
	{
		detect >> inter;
		if (inter.empty())
		{
			cout << "the picture is empty" << endl;
			system("pause");
		};
		vector<Rect>found, found_filtered;
		myHOG.detectMultiScale(inter, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
		//cout << "The number of rectangular boxes found is" << found.size() << endl;
		cout << found.size();
		//Look for all unnested rectangles and put in found_filtered,if has unnested,take the outermost one
		for (unsigned int i = 0; i < found.size(); i++)
		{
			Rect r = found[i];
			unsigned int j = 0;
			for (; j < found.size(); j++)
				if (j != i && (r & found[j]) == r)
					break;
			if (j == found.size())
			{
				found_filtered.push_back(r);
			}
		}
		//drawing rectangular box
		for (unsigned int i = 0; i < found_filtered.size(); i++)
		{
			Rect r = found_filtered[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			rectangle(inter, r.tl(), r.br(), Scalar(0, 255, 0), 3);
			putText(inter, "Drinks", Point2f(r.x + 5, r.y + 10), cv::FONT_HERSHEY_PLAIN, 0.4, Scalar(0, 255, 0), 1, 8, false);
		}
		imshow("output", inter);
		if (waitKey(30)==32)
		{
			break;
		}
	}
	//detect >> inter;
	//resize(inter, frame, Size(64, 128));
	//cvtColor(src, frame, COLOR_BGR2GRAY);
	/*vector<Rect>found, found_filtered;
	myHOG.detectMultiScale(inter, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
	cout << found.size();
	for (unsigned int i = 0; i < found.size(); i++)
	{
		Rect r = found[i];
		unsigned int j = 0;
		for (; j < found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size())
		{
			found_filtered.push_back(r);
		}
	}
	for (unsigned int i = 0; i < found_filtered.size(); i++)
	{
		Rect r = found_filtered[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		rectangle(inter, r.tl(), r.br(), Scalar(0, 255, 0), 3);
		putText(inter, "Drinks", Point2f(r.x + 5, r.y + 10), cv::FONT_HERSHEY_PLAIN, 0.4, Scalar(0, 255, 0), 1, 8, false);
	}
	imshow("output", inter);
	//resize(inter, inter, Size(4 * inter.cols, 4 * inter.rows));
	//imshow("frame", inter);
	//Mat output;
	//resize(src, output, Size(4 * src.cols, 4 * src.rows));
	//imshow("src", output);*/
}

/*利用opencv来对正样本进行拍照use opencv take photos for positive samples*/
void GenerateFileName()
{
	for (size_t i = 0; i < SumPicture; i++)
	{
		ostringstream filename;
		/*stringstream ss;
		string res;
		ss << i;
		ss >> res;*/
		filename << i << Back << endl;
		FILENAME.push_back(filename.str());
	}
}

/*take photos*/
void TakePhotoForPos()
{
	VideoCapture capture(0);
	Mat src;
	if (!capture.isOpened())
	{
		cout << "ERROR" << endl;
		system("pause");
	}
	int PressKeyTime = 0;
	while (true)
	{
		capture >> src;
		imshow("src", src);
		if (waitKey(30)==32)
		{
			ostringstream oss;
			oss << PressKeyTime <<endl;
			imwrite("D:\\traindatabase\\pos\\"+to_string(PressKeyTime)+".jpg", src);
			cout << oss.str() << endl;
			PressKeyTime++;
		}
		if (PressKeyTime>499)
		{
			break;
		}
	}
}

int main(int argc, char** argv)
{
	/*training SVM+HOG*/
	//TakePhotoForPos();
	Train_SVMmodel("D:\\traindatabase\\path.txt", "SVM_HOG.xml");
	Detect_Unit("SVM_HOG.xml");
	//Load_Detect_Vector();
	//Detect_Final();
	return 0;
}
