#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <math.h>

using namespace std;
using namespace cv;

const int cell_rows = 2, cell_cols = 4;
const int p = 4; //缩放倍数
const int rel = 10; //精确计算hog的范围大小

void hog_cal(Mat src, float result[][cell_rows][cell_cols])
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
}

float E_distance(float data1[][cell_rows][cell_cols], float data2[][cell_rows][cell_cols], int x, int y)
{
	float result = 0;
	for (int i = 0; i < 8; ++i)
		result += (data1[i][x][y] - data2[i][x][y]) * (data1[i][x][y] - data2[i][x][y]);
	result = sqrt(result);
	return result;
}

float E_distance_all(float data1[][cell_rows][cell_cols], float data2[][cell_rows][cell_cols])
{
	float result = 0;
	for (int x = 0; x < cell_rows; ++x)
		for (int y = 0; y < cell_cols; ++y)
			result += E_distance(data1, data2, x, y);
	return result;
}

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);
	Mat Template, src;
	Template = imread("../../assets/parts_tracking/parts_Template.png");
	imshow("Template", Template);
	src = imread("../../assets/parts_tracking/parts.png");
	imshow("src", src);

	Mat src_down, Template_down;

	resize(Template, Template_down, Size(Template.cols / p, Template.rows / p));
	resize(src, src_down, Size(src.cols / p, src.rows / p));

	float hog_Template_down[8][cell_rows][cell_cols] = { 0 };
	hog_cal(Template_down, hog_Template_down);

	int src_down_height_end = src_down.rows - Template_down.rows + 1;
	int src_down_width_end = src_down.cols - Template_down.cols + 1;

	float mindis = 123456789;
	int min_x_down = 0, min_y_down = 0;

	for (int i = 0; i < src_down_width_end; ++i)
		for (int j = 0; j < src_down_height_end; ++j)
		{
			Rect rect(i, j, Template_down.cols, Template_down.rows);
			Mat src_down_sub = src_down(rect);

			float hog_src_down[8][cell_rows][cell_cols] = { 0 };
			hog_cal(src_down_sub, hog_src_down);
			float dis = E_distance_all(hog_src_down, hog_Template_down);
			if (mindis > dis)
			{
				mindis = dis;
				min_x_down = i * p;
				min_y_down = j * p;
			}
		}
	printf("x=%d y=%d E_dis=%f\n", min_x_down, min_y_down, mindis);
	
	float hog_Template[8][cell_rows][cell_cols] = { 0 };
	hog_cal(Template, hog_Template);

	int src_height_end = src.rows - Template.rows + 1;
	int src_width_end = src.cols - Template.cols + 1;
	int i_begin = max(0, min_x_down - rel);
	int i_end = min(src_width_end, min_x_down + rel);
	int j_begin = max(0, min_y_down - rel);
	int j_end = min(src_height_end, min_y_down + rel);

	mindis = 123456789;
	int min_x = 0, min_y = 0;
	Rect mindis_rect;
	for (int i = i_begin; i < i_end; ++i)
		for (int j = j_begin; j < j_end; ++j)
		{
			Rect rect(i, j, Template.cols, Template.rows);
			Mat src_sub = src(rect);

			float hog_src[8][cell_rows][cell_cols] = { 0 };
			hog_cal(src_sub, hog_src);
			float dis = E_distance_all(hog_src, hog_Template);
			if (mindis > dis)
			{
				mindis_rect = rect;
				mindis = dis;
				min_x = i;
				min_y = j;
			}
		}
	printf("x=%d y=%d E_dis=%f\n", min_x, min_y, mindis);
	Mat output;
	src.copyTo(output);
	rectangle(output, mindis_rect, Scalar(255, 0, 0), 1, 8, 0);
	imshow("output", output);

	waitKey(0);
}