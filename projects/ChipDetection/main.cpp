#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);

	Mat input = imread("../../assets/chip.png");
	imshow("input", input);

	//灰度
	Mat input_gray;
	cvtColor(input, input_gray, COLOR_BGR2GRAY);
	imshow("input_gray", input_gray);

	//二值化
	Mat thresh_Mat;
	threshold(input_gray, thresh_Mat, 100, 255, THRESH_OTSU);
	imshow("thresh_Mat", thresh_Mat);

	//中值滤波
	Mat blur_Mat;
	medianBlur(thresh_Mat, blur_Mat, 9);
	imshow("blur_Mat", blur_Mat);

	//连通域检测
	Mat labels, stats, centroids;
	int nComp = connectedComponentsWithStats(blur_Mat, labels, stats, centroids, 8, CV_32S);

	//筛选矩形
	Mat output;
	input.copyTo(output);
	for (int i = 1; i < nComp; i++)
	{
		int width = stats.at<int>(i, 2);
		int height = stats.at<int>(i, 3);
		float p = (float)width / height;

		//宽高比例大致满足正方形
		if (p < 0.8 || p > 1.2)
			continue;

		Rect bandbox;
		bandbox.x = stats.at<int>(i, 0);
		bandbox.y = stats.at<int>(i, 1);

		bandbox.width = width;
		bandbox.height = height;
		rectangle(output, bandbox, Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("output", output);

	waitKey(0);
}