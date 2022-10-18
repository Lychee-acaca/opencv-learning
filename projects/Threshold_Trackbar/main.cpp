#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

//回调函数第一个参数th为拖动后滑条位置，第二个参数为传入的img
void onChange(int th, void* data)
{
	Mat src = *(Mat*)(data);
	Mat dst;

	//灰度大于阈值为最大值，其他值为0
	threshold(src, dst, th, 255, THRESH_BINARY);

	//大津法自动寻找全局阈值，因为是自动的所以th没有用，怎么滑滑条都是一样的
	//threshold(src, dst, th, 255, THRESH_OTSU);

	//自适应阈值，在这里仅是列出这个函数，滑条也没有用
	//adaptiveThreshold(src, dst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 15, 10);

	imshow("img", dst);
}

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);

	//输入灰度图
	Mat img = imread("../../assets/grid.jpg", IMREAD_GRAYSCALE);

	//初始阈值设定为0
	Mat init;
	threshold(img, init, 0, 255, 0);
	imshow("img", init);

	/*创建一个滑条
	* 滑条名称为ThresholdTrackbar
	* 创建在img窗口
	* 初始位置0，最大值255
	* 回调函数为onChange，当滑条被拖动会调用回调函数
	* 把img的地址传给回调函数
	*/
	createTrackbar("ThresholdTrackbar", "img", 0, 255, onChange, &img);
	waitKey(0);
}