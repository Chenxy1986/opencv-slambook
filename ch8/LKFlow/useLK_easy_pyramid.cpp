#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
using namespace std; 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

//add by szr


//add by szr
namespace cv
{
void calcOpticalFlowPyrLK_modify( const Mat& prevImg, const Mat& nextImg,
                           const vector<Point2f>& prevPts,
                           vector<Point2f>& nextPts,
                           vector<uchar>& status, vector<float>& err
                            )
{
    //add by szr
    Size winSize = Size(15, 15);
    int maxLevel = 2;
    TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    double derivLambda = 0.0;
    int flags = 0;
    //add by szr
    derivLambda = std::min(std::max(derivLambda, 0.), 1.);
    double lambda1 = 1. - derivLambda, lambda2 = derivLambda;
    const int derivKernelSize = 3;
    const float deriv1Scale = 0.5f/4.f;  //0.125
    const float deriv2Scale = 0.25f/4.f;
    const int derivDepth = CV_32F;  //derivDepth代表类型
    Point2f halfWin((winSize.width-1)*0.5f, (winSize.height-1)*0.5f);
    

    CV_Assert( maxLevel >= 0 && winSize.width > 2 && winSize.height > 2 );
    CV_Assert( prevImg.size() == nextImg.size() &&
        prevImg.type() == nextImg.type() );

    
    size_t npoints = prevPts.size();
    
    nextPts.resize(npoints);
    status.resize(npoints);
    for( size_t i = 0; i < npoints; i++ )
        status[i] = true;
    err.resize(npoints);

    if( npoints == 0 )
        return;
    
    vector<Mat> prevPyr, nextPyr;

    int cn = prevImg.channels();
    //buildPyramid将给定的图像下采样，生成高斯金字塔，金字塔的高度为maxLevel
    buildPyramid( prevImg, prevPyr, maxLevel );
    buildPyramid( nextImg, nextPyr, maxLevel );

    Mat tempDerivIxBuf((prevImg.rows + winSize.height*2),
                  (prevImg.cols + winSize.width*2), 
                  CV_MAKETYPE(derivDepth, cn));
    Mat tempDerivIyBuf((prevImg.rows + winSize.height*2),
                  (prevImg.cols + winSize.width*2), 
                  CV_MAKETYPE(derivDepth, cn));
    
    Mat tempDerivBuf(prevImg.size(), CV_MAKETYPE(derivDepth, cn));

    Mat IBuf((prevImg.rows + winSize.height*2),
                  (prevImg.cols + winSize.width*2), 
                  CV_MAKETYPE(derivDepth, cn));
    Mat JBuf((prevImg.rows + winSize.height*2),
                  (prevImg.cols + winSize.width*2), 
                  CV_MAKETYPE(derivDepth, cn));
    Mat derivIWinBuf(winSize, derivDepth);

    if( (criteria.type & TermCriteria::COUNT) == 0 )
        criteria.maxCount = 30;
    else
        criteria.maxCount = std::min(std::max(criteria.maxCount, 0), 100);
    if( (criteria.type & TermCriteria::EPS) == 0 )
        criteria.epsilon = 0.01;
    else
        criteria.epsilon = std::min(std::max(criteria.epsilon, 0.), 10.);
    criteria.epsilon *= criteria.epsilon;

    //add by szr
    Point2i vector_g;
    vector_g.x = 0;
    vector_g.y = 0;
    for( int level = maxLevel; level >= 0; level-- ){
        Size imgSize = prevPyr[level].size();

    
        Mat _tempDerivIx( imgSize.height + winSize.height*2,
            imgSize.width + winSize.width*2, 
            tempDerivIxBuf.type(), tempDerivIxBuf.data );
        Mat _tempDerivIy( imgSize.height + winSize.height*2,
            imgSize.width + winSize.width*2, 
            tempDerivIyBuf.type(), tempDerivIyBuf.data );
        Mat tempDerivIx(_tempDerivIx, Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
        CvMat cvtempDerivIx = _tempDerivIx;
        cvZero(&cvtempDerivIx);
        Mat tempDerivIy(_tempDerivIy, Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
        CvMat cvtempDerivIy = _tempDerivIy;
        cvZero(&cvtempDerivIy);


        Mat tempDeriv( imgSize, tempDerivBuf.type(), tempDerivBuf.data );

        Mat _tempI( imgSize.height + winSize.height*2,
            imgSize.width + winSize.width*2, 
            IBuf.type(), IBuf.data );
        Mat _tempJ( imgSize.height + winSize.height*2,
            imgSize.width + winSize.width*2, 
            JBuf.type(), JBuf.data );
        Mat tempI(_tempI, Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
        CvMat cvtempI = _tempI;
        cvZero(&cvtempI);
        Mat tempJ(_tempJ, Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
        CvMat cvtempJ = _tempJ;
        cvZero(&cvtempJ);


        vector<int> fromTo(cn*2);
        for(int k = 0; k < cn; k++ )
            fromTo[k*2] = k;
        prevPyr[level].convertTo(tempDeriv, derivDepth);
        for(int k = 0; k < cn; k++ )
            fromTo[k*2+1] = k;
        mixChannels(&tempDeriv, 1, &tempI, 1, &fromTo[0], cn);
        nextPyr[level].convertTo(tempDeriv, derivDepth);
        mixChannels(&tempDeriv, 1, &tempJ, 1, &fromTo[0], cn);
        


        Sobel(prevPyr[level], tempDerivIx, derivDepth, 1, 0, derivKernelSize, deriv1Scale );
        Sobel(prevPyr[level], tempDerivIy, derivDepth, 0, 1, derivKernelSize, deriv1Scale );
        for( size_t ptidx = 0; ptidx < npoints; ptidx++ ){
            Point2f prevPt = prevPts[ptidx]*(float)(1./(1 << level));
            Point2f nextPt;
            if( level == maxLevel )
            {
                if( flags & OPTFLOW_USE_INITIAL_FLOW )
                    nextPt = nextPts[ptidx]*(float)(1./(1 << level));
                else
                    nextPt = prevPt;
            }
            else
                nextPt = nextPts[ptidx]*2.f;
            nextPts[ptidx] = nextPt;

            Point2i iprevPt, inextPt;
            prevPt -= halfWin;
            iprevPt.x = cvFloor(prevPt.x);
            iprevPt.y = cvFloor(prevPt.y);

            if( iprevPt.x < -winSize.width || iprevPt.x >= tempDerivIx.cols ||
                iprevPt.y < -winSize.height || iprevPt.y >= tempDerivIx.rows )
            {
                if( level == 0 )
                {
                    status[ptidx] = false;
                    err[ptidx] = FLT_MAX;
                }
                continue;
            }

            float a = prevPt.x - iprevPt.x;
            float b = prevPt.y - iprevPt.y;
            float w00 = (1.f - a)*(1.f - b), w01 = a*(1.f - b);
            float w10 = (1.f - a)*b, w11 = a*b;

            double iA11 = 0, iA12 = 0, iA22 = 0;


            int x, y;
            const float* src_Ix = (const float*)(tempDerivIx.data);
            const float* src_Iy = (const float*)(tempDerivIy.data);
            int length_y = (tempDerivIx.cols+2*winSize.width)*cn;
            int length_x = cn;
            int index00, index10, index01, index11;
            for( y = 0; y < winSize.height; y++ ){
                for( x = 0; x < winSize.width; x++){
                    index00 = (y+iprevPt.y)*length_y+(x+iprevPt.x)*length_x;
                    index01 = (y+iprevPt.y)*length_y+(x+iprevPt.x+1)*length_x;
                    index10 = (y+iprevPt.y+1)*length_y+(x+iprevPt.x)*length_x;
                    index11 = (y+iprevPt.y+1)*length_y+(x+iprevPt.x+1)*length_x;
                    float Ix = src_Ix[index00]*w00 + src_Ix[index01]*w01 + src_Ix[index10]*w10 + src_Ix[index11]*w11;
                    float Iy = src_Iy[index00]*w00 + src_Iy[index01]*w01 + src_Iy[index10]*w10 + src_Iy[index11]*w11;
                    iA11 += (double)Ix*Ix;
                    iA12 += (double)Ix*Iy;
                    iA22 += (double)Iy*Iy;
                }
            }
            // cout<<"Fin 1"<<endl;

            double D = iA11*iA22 - iA12*iA12;
            double minEig = (iA22 + iA11 - std::sqrt((iA11-iA22)*(iA11-iA22) +
                4.*iA12*iA12))/(2*winSize.width*winSize.height);
            err[ptidx] = (float)minEig;

            if( D < DBL_EPSILON )
            {
                if( level == 0 )
                    status[ptidx] = false;
                continue;
            }
            D = 1./D;
            nextPt -= halfWin;
            Point2f prevDelta;
            for( int j = 0; j < criteria.maxCount; j++ ){
                inextPt.x = cvFloor(nextPt.x);
                inextPt.y = cvFloor(nextPt.y);

                if( inextPt.x < -winSize.width || inextPt.x >= tempDerivIx.cols ||
                    inextPt.y < -winSize.height || inextPt.y >= tempDerivIx.rows )
                {
                    if( level == 0 )
                        status[ptidx] = false;
                    break;
                }

                a = nextPt.x - inextPt.x;
                b = nextPt.y - inextPt.y;
                w00 = (1.f - a)*(1.f - b); w01 = a*(1.f - b);
                w10 = (1.f - a)*b; w11 = a*b;
                double ib1 = 0, ib2 = 0;
                const float* src_I = (const float*)(tempI.data);
                const float* src_J = (const float*)(tempJ.data);
                int Jindex00, Jindex10, Jindex01, Jindex11;
                for( y = 0; y < winSize.height; y++ ){
                    for( x = 0; x < winSize.width; x++){
                        index00 = (y+iprevPt.y)*length_y+(x+iprevPt.x)*length_x;
                        index01 = (y+iprevPt.y)*length_y+(x+iprevPt.x+1)*length_x;
                        index10 = (y+iprevPt.y+1)*length_y+(x+iprevPt.x)*length_x;
                        index11 = (y+iprevPt.y+1)*length_y+(x+iprevPt.x+1)*length_x;
                        float Ix = src_Ix[index00]*w00 + src_Ix[index01]*w01 + src_Ix[index10]*w10 + src_Ix[index11]*w11;
                        float Iy = src_Iy[index00]*w00 + src_Iy[index01]*w01 + src_Iy[index10]*w10 + src_Iy[index11]*w11;
                        
                        Jindex00 = (y+inextPt.y)*length_y+(x+inextPt.x)*length_x;
                        Jindex01 = (y+inextPt.y)*length_y+(1+x+inextPt.x)*length_x;
                        Jindex10 = (1+y+inextPt.y)*length_y+(x+inextPt.x)*length_x;
                        Jindex11 = (1+y+inextPt.y)*length_y+(1+x+inextPt.x)*length_x;
                        float Ik = (-src_I[index00]+src_J[Jindex00])*w00 + (-src_I[index01]+src_J[Jindex01])*w01 + (-src_I[index10]+src_J[Jindex10])*w10 + (-src_I[index11]+src_J[Jindex11])*w11;
                        ib1 += Ix*Ik;
                        ib2 += Iy*Ik;
                    }
                }
                Point2f delta( (float)((iA12*ib2 - iA22*ib1) * D),
                               (float)((iA12*ib1 - iA11*ib2) * D));
                nextPt += delta;
                nextPts[ptidx] = nextPt + halfWin;
                if( delta.ddot(delta) <= criteria.epsilon )
                    break;

                if( j > 0 && std::abs(delta.x + prevDelta.x) < 0.01 &&
                    std::abs(delta.y + prevDelta.y) < 0.01 )
                {
                    nextPts[ptidx] -= delta*0.5f;
                    break;
                }
                prevDelta = delta;
            }
        }
        // cout<<"Fin 2"<<endl;
    }
    // cout<<"Fin 3"<<endl;

    //add by szr
}
}
//add by szr

int main( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"usage: useLK path_to_dataset"<<endl;
        return 1;
    }
    string path_to_dataset = argv[1];
    string associate_file = path_to_dataset + "/associate.txt";
    
    ifstream fin( associate_file );
    if ( !fin ) 
    {
        cerr<<"I cann't find associate.txt!"<<endl;
        return 1;
    }
    
    string rgb_file, depth_file, time_rgb, time_depth;
    list< cv::Point2f > keypoints;      // 因为要删除跟踪失败的点，使用list
    cv::Mat color, depth, last_color;
    
    for ( int index=0; index<100; index++ )
    {
        fin>>time_rgb>>rgb_file>>time_depth>>depth_file;
        color = cv::imread( path_to_dataset+"/"+rgb_file );
        depth = cv::imread( path_to_dataset+"/"+depth_file, -1 );
        if (index ==0 )
        {
            // 对第一帧提取FAST特征点
            vector<cv::KeyPoint> kps;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect( color, kps );
            for ( auto kp:kps )
                keypoints.push_back( kp.pt );
            last_color = color;
            continue;
        }
        if ( color.data==nullptr || depth.data==nullptr )
            continue;
        // 对其他帧用LK跟踪特征点
        vector<cv::Point2f> next_keypoints; 
        vector<cv::Point2f> prev_keypoints;
        for ( auto kp:keypoints )
            prev_keypoints.push_back(kp);
        vector<unsigned char> status;
        vector<float> error; 
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        // cout<<"Start 0"<<endl;
        calcOpticalFlowPyrLK_modify( last_color, color, prev_keypoints, next_keypoints, status, error );
        // cout<<"Fin 0"<<endl;
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
        cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
        // 把跟丢的点删掉
        int i=0; 
        for ( auto iter=keypoints.begin(); iter!=keypoints.end(); i++)
        {
            if ( status[i] == 0 )
            {
                iter = keypoints.erase(iter);
                continue;
            }
            *iter = next_keypoints[i];
            iter++;
        }
        cout<<"tracked keypoints: "<<keypoints.size()<<endl;
        if (keypoints.size() == 0)
        {
            cout<<"all keypoints are lost."<<endl;
            break; 
        }
        // 画出 keypoints
        cv::Mat img_show = color.clone();
        for ( auto kp:keypoints )
            //cv::circle(img_show, kp, 10, cv::Scalar(0, 240, 0), 1);
            cv::rectangle(img_show, cv::Rect((int)kp.x, (int)kp.y ,20, 20), cv::Scalar(0, 240, 0));
        cv::imshow("corners", img_show);
        cv::waitKey(0);
        last_color = color;
    }
    return 0;
}