/*****************************************************************************
*   Number Plate Recognition using SVM and Neural Networks
*/

// Main entry code OpenCV

#include <cv.h>
#include <highgui.h>
#include <cvaux.h>
#include <ml.h>
#include "opencv2/ml.hpp"

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "DetectRegions.h"
#include "OCR.h"

using namespace std;
using namespace cv;

string getFilename(string s) {

    char sep = '/';
    char sepExt='.';

    #ifdef _WIN32
        sep = '\\';
    #endif

    size_t i = s.rfind(sep, s.length( ));
    if (i != string::npos) {
        string fn= (s.substr(i+1, s.length( ) - i));
        size_t j = fn.rfind(sepExt, fn.length( ));
        if (i != string::npos) {
            return fn.substr(0,j);
        }else{
            return fn;
        }
    }else{
        return "";
    }
}

int main ( int argc, char** argv )
{
    cout << "OpenCV Automatic Number Plate Recognition\n";
    char* filename;
    Mat input_image;

    argc=2;
    argv[1]="/Users/wuyongyu/CLionProjects/ANPR/test/2715DTZ.jpg";
    //Check if user specify image to process
    if(argc >= 2 )
    {
        filename= argv[1];
        //load image  in gray level
        input_image=imread(filename,1);
    }else{
        printf("Use:\n\t%s image\n",argv[0]);
        return 0;
    }

    string filename_whithoutExt=getFilename(filename);
    cout << "working with file: "<< filename_whithoutExt << "\n";
    //Detect posibles plate regions
    DetectRegions detectRegions;
    detectRegions.setFilename(filename_whithoutExt);
    detectRegions.saveRegions=true; //TODO:wuyy
    detectRegions.showSteps=false;
    vector<Plate> posible_regions= detectRegions.run( input_image );

    //SVM for each plate region to get valid car plates
    //Read file storage.
    FileStorage fs; //opencv 的filestorage 可以很容易的管理xml或者json 格式的文件
    fs.open("SVM.xml", FileStorage::READ);
    Mat SVM_TrainingData;
    Mat SVM_Classes;
    fs["TrainingData"] >> SVM_TrainingData;
    fs["classes"] >> SVM_Classes;
    //Set SVM params
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);//可以处理非线性分割的问题
    svm->setKernel(cv::ml::SVM::LINEAR);//径向基函数
    svm->setGamma(1);
    svm->setC(1);
    svm->setCoef0(0);
    svm->setDegree(0);
    svm->setNu(0);
    svm->setP(0);
    svm->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER, 1000, 0.01));
//    CvSVMParams SVM_params;
//    SVM_params.svm_type = CvSVM::C_SVC;
//    SVM_params.kernel_type = CvSVM::LINEAR; //CvSVM::LINEAR;  核函数
//    SVM_params.degree = 0;
//    SVM_params.gamma = 1;
//    SVM_params.coef0 = 0;
//    SVM_params.C = 1;
//    SVM_params.nu = 0;
//    SVM_params.p = 0;
//    SVM_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01);
    //Train SVM

//    CvSVM svmClassifier(SVM_TrainingData, SVM_Classes, Mat(), Mat(), SVM_params);
    svm->train(SVM_TrainingData, ml::SampleTypes::ROW_SAMPLE,SVM_Classes);

    //For each possible plate, classify with svm if it's a plate or no
    vector<Plate> plates;
    for(int i=0; i< posible_regions.size(); i++)
    {
        Mat img=posible_regions[i].plateImg;
        Mat p= img.reshape(1, 1);
        p.convertTo(p, CV_32FC1);

        int response = (int)svm->predict( p );  //使用svm 预测是不是车牌
        if(response==1)
            plates.push_back(posible_regions[i]);
    }

    cout << "Num plates detected: " << plates.size() << "\n";
    //For each plate detected, recognize it with OCR
    OCR ocr("OCR.xml");
    ocr.saveSegments=true;
    ocr.DEBUG=false;
    ocr.filename=filename_whithoutExt;
    for(int i=0; i< plates.size(); i++){
        Plate plate=plates[i];

        string plateNumber=ocr.run(&plate);
        string licensePlate=plate.str();
        cout << "================================================\n";
        cout << "License plate number: "<< licensePlate << "\n";
        cout << "================================================\n";
        rectangle(input_image, plate.position, Scalar(0,0,200));
        putText(input_image, licensePlate, Point(plate.position.x, plate.position.y), CV_FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,200),2);
        if(false){
            imshow("Plate Detected seg", plate.plateImg);
            cvWaitKey(0);
        }

    }
        imshow("Plate Detected", input_image);
       for(;;)
       {
       int c;
       c = cvWaitKey(10);
       if( (char) c == 27)
       break;
       }
    return 0;
}
