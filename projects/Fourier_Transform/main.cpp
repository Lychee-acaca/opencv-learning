#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

int ifftDemo();
void on_mouse(int EVENT, int x, int y, int flags, void* userdata);
int selectPolygon(cv::Mat srcMat, cv::Mat& dstMat);

vector<Point> mousePoints;
Point points;

int ifftDemo()
{
    cv::Mat dst;
    cv::Mat src = imread("../../assets/lena.jpg", 0);

    int m = getOptimalDFTSize(src.rows); //2,3,5的倍数有更高效率的傅里叶变换
    int n = getOptimalDFTSize(src.cols);
    Mat padded;
    //把灰度图像放在左上角,在右边和下边扩展图像,扩展部分填充为0;
    copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));
    //planes[0]为dft变换的实部，planes[1]为虚部，ph为相位， plane_true=mag为幅值
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat planes_true = Mat_<float>(padded);
    Mat ph = Mat_<float>(padded);
    Mat complexImg;
    //多通道complexImg既有实部又有虚部
    merge(planes, 2, complexImg);
    //对上边合成的mat进行傅里叶变换,***支持原地操作***,傅里叶变换结果为复数.通道1存的是实部,通道二存的是虚部
    dft(complexImg, complexImg);
    //把变换后的结果分割到两个mat,一个实部,一个虚部,方便后续操作
    split(complexImg, planes);

    //---------------此部分目的为更好地显示幅值---后续恢复原图时反着再处理一遍-------------------------
    magnitude(planes[0], planes[1], planes_true);//幅度谱mag
    phase(planes[0], planes[1], ph);//相位谱ph
    Mat A = planes[0];
    Mat B = planes[1];
    Mat mag = planes_true;

    mag += Scalar::all(1);//对幅值加1
    //计算出的幅值一般很大，达到10^4,通常没有办法在图像中显示出来，需要对其进行log求解。
    log(mag, mag);

    //取矩阵中的最大值，便于后续还原时去归一化
    double maxVal;
    minMaxLoc(mag, 0, &maxVal, 0, 0);

    //修剪频谱,如果图像的行或者列是奇数的话,那其频谱是不对称的,因此要修剪
    mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));
    ph = ph(Rect(0, 0, mag.cols & -2, mag.rows & -2));
    Mat _magI = mag.clone();
    //将幅度归一化到可显示范围。
    normalize(_magI, _magI, 0, 1, NORM_MINMAX);
    //imshow("before rearrange", _magI);

    //显示规则频谱图
    int cx = mag.cols / 2;
    int cy = mag.rows / 2;

    //这里是以中心为标准，把mag图像分成四部分
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
    //imshow("原图灰度图", src);
    //imshow("频谱幅度", mag);
    mag = mag * 255;
    //imwrite("原频谱.jpg", mag);
    /*--------------------------------------------------*/

    mag = mag / 255;
    cv::Mat mask;
    Mat proceMag;

    selectPolygon(mag, mask);

    mag = mag.mul(mask);

    proceMag = mag * 255;
    //imwrite("处理后频谱.jpg", proceMag);
    imshow("处理后频谱", mag);

    //前述步骤反着来一遍，目的是为了逆变换回原图
    Mat q00(mag, Rect(0, 0, cx, cy));
    Mat q10(mag, Rect(cx, 0, cx, cy));
    Mat q20(mag, Rect(0, cy, cx, cy));
    Mat q30(mag, Rect(cx, cy, cx, cy));

    //交换象限
    q00.copyTo(tmp);
    q30.copyTo(q00);
    tmp.copyTo(q30);
    q10.copyTo(tmp);
    q20.copyTo(q10);
    tmp.copyTo(q20);

    mag = mag * maxVal;//将归一化的矩阵还原
    exp(mag, mag);//对应于前述去对数
    mag = mag - Scalar::all(1);//对应前述+1
    polarToCart(mag, ph, planes[0], planes[1]);//由幅度谱mag和相位谱ph恢复实部planes[0]和虚部planes[1]
    merge(planes, 2, complexImg);//将实部虚部合并


    //-----------------------傅里叶的逆变换-----------------------------------
    Mat ifft(Size(src.cols, src.rows), CV_8UC1);
    //傅里叶逆变换
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

void on_mouse(int EVENT, int x, int y, int flags, void* userdata)
{
    Mat hh;
    hh = *(Mat*)userdata;
    Point p(x, y);
    if (EVENT == EVENT_LBUTTONDOWN)
    {
        points.x = x;
        points.y = y;
        mousePoints.push_back(points);
        circle(hh, points, 1, Scalar(255, 255, 255), -1);
        imshow("mouseCallback", hh);
    }
}

int selectPolygon(cv::Mat srcMat, cv::Mat& dstMat)
{
    vector<vector<Point>> contours;
    cv::Mat selectMat;

    cv::Mat m = cv::Mat::zeros(srcMat.size(), CV_32F);

    m = 1;

    if (!srcMat.empty()) {
        srcMat.copyTo(selectMat);
        srcMat.copyTo(dstMat);
    }
    else {
        std::cout << "failed to read image!:" << std::endl;
        return -1;
    }

    namedWindow("mouseCallback");
    imshow("mouseCallback", selectMat);
    setMouseCallback("mouseCallback", on_mouse, &selectMat);
    waitKey(0);
    destroyAllWindows();
    //计算roi
    contours.push_back(mousePoints);
    if (contours[0].size() < 3) {
        std::cout << "failed to read image!:" << std::endl;
        return -1;
    }

    drawContours(m, contours, 0, Scalar(0), -1);

    m.copyTo(dstMat);

    return 0;
}

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);
    ifftDemo();
}