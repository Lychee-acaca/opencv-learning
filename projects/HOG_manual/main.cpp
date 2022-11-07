#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <math.h>

using namespace std;
using namespace cv;

void hog_cal(Mat src, float result[][8][3], int cell_rows, int cell_cols)
{
	Mat src_gray;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	Mat gx, gy;
	Sobel(src_gray, gx, CV_32F, 1, 0, 1);
	Sobel(src_gray, gy, CV_32F, 0, 1, 1);

	Mat mag, angel;
	cartToPolar(gx, gy, mag, angel, true);

	int cell_height = src.rows / cell_rows;
	int cell_width = src.cols / cell_cols;

	for (int x = 0; x < cell_rows; ++x)
		for (int y = 0; y < cell_cols; ++y)
			for (int i = 0; i < cell_height; ++i)
				for (int j = 0; j < cell_width; ++j)
				{
					int p = angel.at<float>(x * cell_height + i, y * cell_width + j) / 45;
					result[p][x][y] += mag.at<float>(i, j);
				}

	/*
	for (int i = 0; i < src.rows; ++i)
		for (int j = 0; j < src.cols; ++j)
		{
			int p = angel.at<float>(i, j) / 45;
			result[p] += mag.at<float>(i, j);
		}
	*/
}

void drawHist(float data[][8][3], const String& winname, int x, int y)
{
	Mat hist(Size(400, 300), CV_8UC3, Scalar(0));

	for (int i = 0; i < 8; ++i)
	{
		rectangle(hist, 
			Point(i * 50, 299), Point((i + 1) * 50 - 1, 299 - (data[i][x][y] / 100000 * 300)),
			Scalar(255 - i * 30, i * 30, 255 - i * 30), -1, 8, 0);
	}
	imshow(winname, hist);
}

float E_distance(float data1[][8][3], float data2[][8][3], int x, int y)
{
	float result = 0;
	for (int i = 0; i < 8; ++i)
		result += (data1[i][x][y] - data2[i][x][y]) * (data1[i][x][y] - data2[i][x][y]);
	result = sqrt(result);
	return result;
}

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);
	Mat Template, img1, img2;
	Template = imread("../../assets/hog/hogTemplate.jpg");
	imshow("Template", Template);
	img1 = imread("../../assets/hog/img1.jpg");
	imshow("img1", img1);
	img2 = imread("../../assets/hog/img2.jpg");
	imshow("img2", img2);

	//int cell_rows = 1, cell_cols = 1;
	int cell_rows = 8, cell_cols = 3;

	float hog_Template[8][8][3] = {0};
	float hog_img1[8][8][3] = { 0 };
	float hog_img2[8][8][3] = { 0 };
	hog_cal(Template, hog_Template, cell_rows, cell_cols);
	hog_cal(img1, hog_img1, cell_rows, cell_cols);
	hog_cal(img2, hog_img2, cell_rows, cell_cols);

	/*
	for (int i = 0; i < 8; ++i)
		printf("%f %f %f\n", hog_Template[i], hog_img1[i], hog_img2[i]);
	*/

	int test_x = 5, test_y = 1;

	drawHist(hog_Template, "Template_hist", test_x, test_y);
	drawHist(hog_img1, "img1_hist", test_x, test_y);
	drawHist(hog_img2, "img2_hist", test_x, test_y);

	printf("%f\n%f\n%f\n", 
		E_distance(hog_Template, hog_img1, test_x, test_y),
		E_distance(hog_Template, hog_img2, test_x, test_y),
		E_distance(hog_img1, hog_img2, test_x, test_y));
	
	waitKey(0);
}