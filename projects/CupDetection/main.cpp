#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

//画hsv直方图分析杯盖，h色调用蓝色表示，s饱和度用绿色表示，v亮度用红色表示
Mat draw_HSV_Hist(Mat& image_hsv)
{
	vector<Mat> hsv;
	split(image_hsv, hsv);

	int numbins = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	Mat h_hist, s_hist, v_hist;
	calcHist(&hsv[0], 1, 0, Mat(), h_hist, 1, &numbins, &histRange);
	calcHist(&hsv[1], 1, 0, Mat(), s_hist, 1, &numbins, &histRange);
	calcHist(&hsv[2], 1, 0, Mat(), v_hist, 1, &numbins, &histRange);

	int width = 512;
	int height = 300;
	Mat histImage(height, width, CV_8UC3, Scalar(20, 20, 20));

	//将图像直方图的高度标准化为与输出直方图的高度一样
	normalize(h_hist, h_hist, 0, height, NORM_MINMAX);
	normalize(s_hist, s_hist, 0, height, NORM_MINMAX);
	normalize(v_hist, v_hist, 0, height, NORM_MINMAX);

	int binStep = cvRound((float)width / (float)numbins);
	for (int i = 1; i < numbins; i++)
	{
		line(histImage, Point(binStep * (i - 1), height - cvRound(h_hist.at<float>(i - 1))),
			Point(binStep * (i), height - cvRound(h_hist.at<float>(i))),
			Scalar(255, 0, 0));
		line(histImage, Point(binStep * (i - 1), height - cvRound(s_hist.at<float>(i - 1))),
			Point(binStep * (i), height - cvRound(s_hist.at<float>(i))),
			Scalar(0, 255, 0));
		line(histImage, Point(binStep * (i - 1), height - cvRound(v_hist.at<float>(i - 1))),
			Point(binStep * (i), height - cvRound(v_hist.at<float>(i))),
			Scalar(0, 0, 255));
		//printf("h: %d %d\n", i, cvRound(h_hist.at<float>(i - 1)));
		//printf("s: %d %d\n", i, cvRound(s_hist.at<float>(i - 1)));
		//printf("v: %d %d\n", i, cvRound(v_hist.at<float>(i - 1)));
	}
	return histImage;
}

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);

	Mat input = imread("../../assets/cup.jpg");
	imshow("input", input);

	Mat input_hsv;
	cvtColor(input, input_hsv, COLOR_BGR2HSV);

	//画hsv直方图
	Mat hist = draw_HSV_Hist(input_hsv);
	imshow("histImage", hist);

	//筛选颜色
	Mat maskMat;
	double i_minH = 176, i_maxH = 183;
	double i_minS = 127, i_maxS = 204;
	double i_minV = 110, i_maxV = 250;
	inRange(input_hsv, Scalar(i_minH, i_minS, i_minV), Scalar(i_maxH, i_maxS, i_maxV), maskMat);
	imshow("maskMat", maskMat);

	//开运算
	Mat maskMat_open;
	morphologyEx(maskMat, maskMat_open, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));
	imshow("maskMat_open", maskMat_open);

	//闭运算
	Mat maskMat_close;
	morphologyEx(maskMat_open, maskMat_close, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(9, 9)));
	imshow("maskMat_close", maskMat_close);

	//连通域检测
	Mat labels, stats, centroids;
	int nComp = connectedComponentsWithStats(maskMat_close, labels, stats, centroids, 8, CV_32S);

	Mat output;
	input.copyTo(output);
	//画矩形
	for (int i = 1; i < nComp; i++)
	{
		int width = stats.at<int>(i, 2);
		int height = stats.at<int>(i, 3);

		Rect bandbox;
		bandbox.x = stats.at<int>(i, 0);
		bandbox.y = stats.at<int>(i, 1);

		bandbox.width = stats.at<int>(i, 2);
		bandbox.height = stats.at<int>(i, 3);
		rectangle(output, bandbox, 255, 1, 8, 0);
	}

	imshow("output", output);
	waitKey(0);
}