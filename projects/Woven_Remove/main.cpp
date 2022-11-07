#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

int ifftDemo();

vector<Point> mousePoints;
Point points;

int ifftDemo()
{
    cv::Mat dst;
    cv::Mat src = imread("../../assets/woven.jpg", 0);

    int m = getOptimalDFTSize(src.rows); //2,3,5�ı����и���Ч�ʵĸ���Ҷ�任
    int n = getOptimalDFTSize(src.cols);
    Mat padded;
    //�ѻҶ�ͼ��������Ͻ�,���ұߺ��±���չͼ��,��չ�������Ϊ0;
    copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));
    //planes[0]Ϊdft�任��ʵ����planes[1]Ϊ�鲿��phΪ��λ�� plane_true=magΪ��ֵ
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat planes_true = Mat_<float>(padded);
    Mat ph = Mat_<float>(padded);
    Mat complexImg;
    //��ͨ��complexImg����ʵ�������鲿
    merge(planes, 2, complexImg);
    //���ϱߺϳɵ�mat���и���Ҷ�任,***֧��ԭ�ز���***,����Ҷ�任���Ϊ����.ͨ��1�����ʵ��,ͨ����������鲿
    dft(complexImg, complexImg);
    //�ѱ任��Ľ���ָ����mat,һ��ʵ��,һ���鲿,�����������
    split(complexImg, planes);

    //---------------�˲���Ŀ��Ϊ���õ���ʾ��ֵ---�����ָ�ԭͼʱ�����ٴ���һ��-------------------------
    magnitude(planes[0], planes[1], planes_true);//������mag
    phase(planes[0], planes[1], ph);//��λ��ph
    Mat A = planes[0];
    Mat B = planes[1];
    Mat mag = planes_true;

    mag += Scalar::all(1);//�Է�ֵ��1
    //������ķ�ֵһ��ܴ󣬴ﵽ10^4,ͨ��û�а취��ͼ������ʾ��������Ҫ�������log��⡣
    log(mag, mag);

    //ȡ�����е����ֵ�����ں�����ԭʱȥ��һ��
    double maxVal;
    minMaxLoc(mag, 0, &maxVal, 0, 0);

    //�޼�Ƶ��,���ͼ����л������������Ļ�,����Ƶ���ǲ��ԳƵ�,���Ҫ�޼�
    mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));
    ph = ph(Rect(0, 0, mag.cols & -2, mag.rows & -2));
    Mat _magI = mag.clone();
    //�����ȹ�һ��������ʾ��Χ��
    normalize(_magI, _magI, 0, 1, NORM_MINMAX);
    //imshow("before rearrange", _magI);

    //��ʾ����Ƶ��ͼ
    int cx = mag.cols / 2;
    int cy = mag.rows / 2;

    //������������Ϊ��׼����magͼ��ֳ��Ĳ���
    Mat tmp;
    Mat q0(mag, Rect(0, 0, cx, cy));
    Mat q1(mag, Rect(cx, 0, cx, cy));
    Mat q2(mag, Rect(0, cy, cx, cy));
    Mat q3(mag, Rect(cx, cy, cx, cy));
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(mag, mag, 0, 1, NORM_MINMAX);
    imshow("mag����ǰ", mag);

    Mat mag_2;
    mag.copyTo(mag_2);
    mag_2 = mag_2 * 255;
    mag_2.convertTo(mag_2, CV_8U);

    //imshow("1", mag_2);

    //��ֵ�˲�
    medianBlur(mag_2, mag_2, 11);
    //imshow("2", mag_2);

    //���
    threshold(mag_2, mag_2, 0, 255, THRESH_OTSU);
    //imshow("3", mag_2);

    //������
    morphologyEx(mag_2, mag_2, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(7, 7)));
    //imshow("4", mag_2);

    //Ѱ������
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(mag_2, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    //ɸѡ
    int num = 0;
    for (int i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]); //�����������
        if (area > 100 && area < 100000)
        {
            num++;
            drawContours(mag, contours, i, Scalar(0), -1);
        }
    }
    imshow("mag�����", mag);


    //ǰ�����跴����һ�飬Ŀ����Ϊ����任��ԭͼ
    Mat q00(mag, Rect(0, 0, cx, cy));
    Mat q10(mag, Rect(cx, 0, cx, cy));
    Mat q20(mag, Rect(0, cy, cx, cy));
    Mat q30(mag, Rect(cx, cy, cx, cy));

    //��������
    q00.copyTo(tmp);
    q30.copyTo(q00);
    tmp.copyTo(q30);
    q10.copyTo(tmp);
    q20.copyTo(q10);
    tmp.copyTo(q20);

    mag = mag * maxVal;//����һ���ľ���ԭ
    exp(mag, mag);//��Ӧ��ǰ��ȥ����
    mag = mag - Scalar::all(1);//��Ӧǰ��+1
    polarToCart(mag, ph, planes[0], planes[1]);//�ɷ�����mag����λ��ph�ָ�ʵ��planes[0]���鲿planes[1]
    merge(planes, 2, complexImg);//��ʵ���鲿�ϲ�


    //-----------------------����Ҷ����任-----------------------------------
    Mat ifft(Size(src.cols, src.rows), CV_8UC1);
    //����Ҷ��任
    idft(complexImg, ifft, DFT_REAL_OUTPUT);
    normalize(ifft, ifft, 0, 1, NORM_MINMAX);

    Rect rect(0, 0, src.cols, src.rows);
    dst = ifft(rect);
    dst = dst * 255;

    cv::Mat dspMat;
    dst.convertTo(dspMat, CV_8UC1);
    imshow("dst", dspMat);
    imshow("src", src);
    waitKey(0);

    return 0;
}

int main()
{
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);
    ifftDemo();
}