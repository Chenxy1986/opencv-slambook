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
    // I, dI/dx ~ Ix, dI/dy ~ Iy, d2I/dx2 ~ Ixx, d2I/dxdy ~ Ixy, d2I/dy2 ~ Iyy
    Mat derivIBuf((prevImg.rows + winSize.height*2),
                  (prevImg.cols + winSize.width*2),
                  CV_MAKETYPE(derivDepth, cn*6));
    // J, dJ/dx ~ Jx, dJ/dy ~ Jy
    Mat derivJBuf((prevImg.rows + winSize.height*2),
                  (prevImg.cols + winSize.width*2),
                  CV_MAKETYPE(derivDepth, cn*3));
    Mat tempDerivBuf(prevImg.size(), CV_MAKETYPE(derivIBuf.type(), cn));
    Mat derivIWinBuf(winSize, derivIBuf.type());

    if( (criteria.type & TermCriteria::COUNT) == 0 )
        criteria.maxCount = 30;
    else
        criteria.maxCount = std::min(std::max(criteria.maxCount, 0), 100);
    if( (criteria.type & TermCriteria::EPS) == 0 )
        criteria.epsilon = 0.01;
    else
        criteria.epsilon = std::min(std::max(criteria.epsilon, 0.), 10.);
    criteria.epsilon *= criteria.epsilon;

    //这里开始遍历每层高斯金字塔
    for( int level = maxLevel; level >= 0; level-- )
    {
        int k;
        Size imgSize = prevPyr[level].size();
        
        Mat tempDeriv( imgSize, tempDerivBuf.type(), tempDerivBuf.data );
        Mat _derivI( imgSize.height + winSize.height*2,
            imgSize.width + winSize.width*2,
            derivIBuf.type(), derivIBuf.data );
        // std::cout<<imgSize<<std::endl;
        // std::cout<<derivIBuf.type()<<std::endl;  //141
        // std::cout<<CV_ELEM_SIZE(derivIBuf.type())<<std::endl;  //72
        // std::cout<<_derivI.step<<std::endl;
        
            
        Mat _derivJ( imgSize.height + winSize.height*2,
            imgSize.width + winSize.width*2,
            derivJBuf.type(), derivJBuf.data );
        Mat derivI(_derivI, Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
        Mat derivJ(_derivJ, Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
        CvMat cvderivI = _derivI;
        cvZero(&cvderivI);
        CvMat cvderivJ = _derivJ;
        cvZero(&cvderivJ);

        vector<int> fromTo(cn*2);
        for( k = 0; k < cn; k++ )
            fromTo[k*2] = k; //fromTo被mixChannel使用
        for(int i=0;i<6;i++)
        {
            cout<<fromTo[i]<<" ";
        }
        cout<<endl;

        prevPyr[level].convertTo(tempDeriv, derivDepth);
        for( k = 0; k < cn; k++ )
            fromTo[k*2+1] = k*6;

        //mixChannel的作用是将tempDeriv中的数据复制给derivI，具体复制到derivI的哪里是根据fromTo指定的
  
        mixChannels(&tempDeriv, 1, &derivI, 1, &fromTo[0], cn);

        // compute spatial derivatives and merge them together
        //Sobel的作用是对tempDeriv的部分点（prevPyr[level]）求梯度；1,0表示求x方向的一阶梯度，0,1表示求y方向的一阶梯度，1,1表示对x、y方向的一阶梯度
        Sobel(prevPyr[level], tempDeriv, derivDepth, 1, 0, derivKernelSize, deriv1Scale );
        
        // int aaaa[9]={1,0,0,0,0,0,0, 0,0};
        // size_t a_size = 9;
        // Mat sobel_input(a_size,CV_8UC1,aaaa);

        for( k = 0; k < cn; k++ )
            fromTo[k*2+1] = k*6 + 1;
        mixChannels(&tempDeriv, 1, &derivI, 1, &fromTo[0], cn);

        Sobel(prevPyr[level], tempDeriv, derivDepth, 0, 1, derivKernelSize, deriv1Scale );
        for( k = 0; k < cn; k++ )
            fromTo[k*2+1] = k*6 + 2;
        mixChannels(&tempDeriv, 1, &derivI, 1, &fromTo[0], cn);
  
        Sobel(prevPyr[level], tempDeriv, derivDepth, 2, 0, derivKernelSize, deriv2Scale );
        for( k = 0; k < cn; k++ )
            fromTo[k*2+1] = k*6 + 3;
        mixChannels(&tempDeriv, 1, &derivI, 1, &fromTo[0], cn);

        Sobel(prevPyr[level], tempDeriv, derivDepth, 1, 1, derivKernelSize, deriv2Scale );
        for( k = 0; k < cn; k++ )
            fromTo[k*2+1] = k*6 + 4;
        mixChannels(&tempDeriv, 1, &derivI, 1, &fromTo[0], cn);

        Sobel(prevPyr[level], tempDeriv, derivDepth, 0, 2, derivKernelSize, deriv2Scale );
        for( k = 0; k < cn; k++ )
            fromTo[k*2+1] = k*6 + 5;
        mixChannels(&tempDeriv, 1, &derivI, 1, &fromTo[0], cn);

        nextPyr[level].convertTo(tempDeriv, derivDepth);
        for( k = 0; k < cn; k++ )
            fromTo[k*2+1] = k*3;
        mixChannels(&tempDeriv, 1, &derivJ, 1, &fromTo[0], cn);

        Sobel(nextPyr[level], tempDeriv, derivDepth, 1, 0, derivKernelSize, deriv1Scale );
        for( k = 0; k < cn; k++ )
            fromTo[k*2+1] = k*3 + 1;
        mixChannels(&tempDeriv, 1, &derivJ, 1, &fromTo[0], cn);

        Sobel(nextPyr[level], tempDeriv, derivDepth, 0, 1, derivKernelSize, deriv1Scale );
        for( k = 0; k < cn; k++ )
            fromTo[k*2+1] = k*3 + 2;
        mixChannels(&tempDeriv, 1, &derivJ, 1, &fromTo[0], cn);

        /*copyMakeBorder( derivI, _derivI, winSize.height, winSize.height,
            winSize.width, winSize.width, BORDER_CONSTANT );
        copyMakeBorder( derivJ, _derivJ, winSize.height, winSize.height,
            winSize.width, winSize.width, BORDER_CONSTANT );*/

        //遍历每个点
        for( size_t ptidx = 0; ptidx < npoints; ptidx++ )
        {
            float scale_level; //0.25,0.5,1
            scale_level = (1./(1 << level));
            Point2f prevPt = prevPts[ptidx]*scale_level;
            Point2f nextPt;
            if( level == maxLevel )
            {
                if( flags & OPTFLOW_USE_INITIAL_FLOW ) //OPTFLOW_USE_INITIAL_FLOW = 4
                    nextPt = nextPts[ptidx]*(float)(1./(1 << level));
                else
                    nextPt = prevPt;
                    
            }
            else
                nextPt = nextPts[ptidx]*2.f;
            nextPts[ptidx] = nextPt;
            
            Point2i iprevPt, inextPt;
            prevPt -= halfWin; //halfWin=(7,7)
            iprevPt.x = cvFloor(prevPt.x);
            iprevPt.y = cvFloor(prevPt.y);

            if( iprevPt.x < -winSize.width || iprevPt.x >= derivI.cols ||
                iprevPt.y < -winSize.height || iprevPt.y >= derivI.rows )
            {
                if( level == 0 )
                {
                    status[ptidx] = false;
                    err[ptidx] = FLT_MAX;
                }
                continue;
            }


            //prevPt.x不知道为什么是浮点型的数据，我觉得prevPt.x应该是定点整数，因为prevPt.x表示该点在图像中的x坐标位置
            float a = prevPt.x - iprevPt.x;
            float b = prevPt.y - iprevPt.y;
            float w00 = (1.f - a)*(1.f - b), w01 = a*(1.f - b);
            float w10 = (1.f - a)*b, w11 = a*b;
            size_t stepI = derivI.step/derivI.elemSize1();
            size_t stepJ = derivJ.step/derivJ.elemSize1();
            // int ffff = ptidx;
            // if(ffff == 0){
            //     std::cout<<stepI<<std::endl;
            //     std::cout<<stepJ<<std::endl;
            //     std::cout<<derivI.step<<std::endl;
            //      std::cout<<derivI.elemSize1()<<std::endl;
            // }
            //cnI和cnJ用于读取刚才求出的对I和J（原始帧和对比帧）的梯度
            int cnI = cn*6, cnJ = cn*3;
            double A11 = 0, A12 = 0, A22 = 0;
            double iA11 = 0, iA12 = 0, iA22 = 0;
            
            // extract the patch from the first image
            int x, y;
            //详见光流算法的公式10，需要将window中的梯度累加
            for( y = 0; y < winSize.height; y++ )
            {
                const float* src = (const float*)(derivI.data +
                    (y + iprevPt.y)*derivI.step) + iprevPt.x*cnI;  //derivI.step=cols * esz;(670*72)
                // cout<<"prevPt.x "<<prevPt.x<<" y + iprevPt.y "<<y + iprevPt.y<<endl;
                // cout<<"index "<<(int)((y + iprevPt.y)*derivI.step)<<endl;
                // getchar();

                float* dst = (float*)(derivIWinBuf.data + y*derivIWinBuf.step);

                for( x = 0; x < winSize.width*cnI; x += cnI, src += cnI )
                {
                    float I = src[0]*w00 + src[cnI]*w01 + src[stepI]*w10 + src[stepI+cnI]*w11;
                    dst[x] = I;
                    
                    float Ix = src[1]*w00 + src[cnI+1]*w01 + src[stepI+1]*w10 + src[stepI+cnI+1]*w11;
                    float Iy = src[2]*w00 + src[cnI+2]*w01 + src[stepI+2]*w10 + src[stepI+cnI+2]*w11;
                    dst[x+1] = Ix; dst[x+2] = Iy;
                    
                    float Ixx = src[3]*w00 + src[cnI+3]*w01 + src[stepI+3]*w10 + src[stepI+cnI+3]*w11;
                    float Ixy = src[4]*w00 + src[cnI+4]*w01 + src[stepI+4]*w10 + src[stepI+cnI+4]*w11;
                    float Iyy = src[5]*w00 + src[cnI+5]*w01 + src[stepI+5]*w10 + src[stepI+cnI+5]*w11;
                    dst[x+3] = Ixx; dst[x+4] = Ixy; dst[x+5] = Iyy;

                    iA11 += (double)Ix*Ix;
                    iA12 += (double)Ix*Iy;
                    iA22 += (double)Iy*Iy;

                    A11 += (double)Ixx*Ixx + (double)Ixy*Ixy;
                    A12 += Ixy*((double)Ixx + Iyy);
                    A22 += (double)Ixy*Ixy + (double)Iyy*Iyy;
                }
            }

            A11 = lambda1*iA11 + lambda2*A11;
            A12 = lambda1*iA12 + lambda2*A12;
            A22 = lambda1*iA22 + lambda2*A22;

            //D对应光流算法的G
            double D = A11*A22 - A12*A12;
            double minEig = (A22 + A11 - std::sqrt((A11-A22)*(A11-A22) +
                4.*A12*A12))/(2*winSize.width*winSize.height);
            err[ptidx] = (float)minEig;

            if( D < DBL_EPSILON )
            {
                if( level == 0 )
                    status[ptidx] = false;
                continue;
            }
            
            //求出二维矩阵G的逆
            D = 1./D;
            

            nextPt -= halfWin;
            Point2f prevDelta;

            for( int j = 0; j < criteria.maxCount; j++ )
            {
                inextPt.x = cvFloor(nextPt.x);
                inextPt.y = cvFloor(nextPt.y);

                if( inextPt.x < -winSize.width || inextPt.x >= derivJ.cols ||
                    inextPt.y < -winSize.height || inextPt.y >= derivJ.rows )
                {
                    if( level == 0 )
                        status[ptidx] = false;
                    break;
                }

                a = nextPt.x - inextPt.x;
                b = nextPt.y - inextPt.y;
                w00 = (1.f - a)*(1.f - b); w01 = a*(1.f - b);
                w10 = (1.f - a)*b; w11 = a*b;

                double b1 = 0, b2 = 0, ib1 = 0, ib2 = 0;

                for( y = 0; y < winSize.height; y++ )
                {
                    const float* src = (const float*)(derivJ.data +
                        (y + inextPt.y)*derivJ.step) + inextPt.x*cnJ;
                        cout<<"derivI.step "<<derivI.step<<endl;
                        cout<<"derivJ.step "<<derivJ.step<<endl;
                        cout<<"derivJ.cols "<<derivJ.cols<<endl;
                        getchar();
                    const float* Ibuf = (float*)(derivIWinBuf.data + y*derivIWinBuf.step);

                    for( x = 0; x < winSize.width; x++, src += cnJ, Ibuf += cnI )
                    {
                        double It = src[0]*w00 + src[cnJ]*w01 + src[stepJ]*w10 +
                                    src[stepJ+cnJ]*w11 - Ibuf[0];
                        //double Ixt = src[1]*w00 + src[cnJ+1]*w01 + src[stepJ+1]*w10 +
                                     //src[stepJ+cnJ+1]*w11 - Ibuf[1];
                        //double Iyt = src[2]*w00 + src[cnJ+2]*w01 + src[stepJ+2]*w10 +
                                     //src[stepJ+cnJ+2]*w11 - Ibuf[2];
                        //b1 += Ixt*Ibuf[3] + Iyt*Ibuf[4];
                        //b2 += Ixt*Ibuf[4] + Iyt*Ibuf[5];
                        ib1 += It*Ibuf[1];
                        ib2 += It*Ibuf[2];
                    }
                }

                b1 = lambda1*ib1 + lambda2*b1;
                b2 = lambda1*ib2 + lambda2*b2;
                Point2f delta( (float)((A12*b2 - A22*b1) * D),
                               (float)((A12*b1 - A11*b2) * D));
                //delta = -delta;

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
    }
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
        calcOpticalFlowPyrLK_modify( last_color, color, prev_keypoints, next_keypoints, status, error );
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