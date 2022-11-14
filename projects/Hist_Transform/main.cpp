#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);

	Mat input = imread("../../assets/lena.jpg");
	Mat input_gray;
	cvtColor(input, input_gray, COLOR_BGR2GRAY);

	imshow("input", input_gray);

	int h = input_gray.rows;
	int w = input_gray.cols;
	
	//线性变换
	uchar LUT_line[256];
	uchar lim = 0.4 * 256;
	for (int i = 0; i < 256; ++i)
		if (i < lim)
			LUT_line[i] = i * (1 / 0.4);
		else
			LUT_line[i] = 1;

	Mat output_line_gray;
	input_gray.copyTo(output_line_gray);
	for (int i = 0; i < h; ++i)
		for (int j = 0; j < w; ++j)
			output_line_gray.at<uchar>(i, j) = LUT_line[input_gray.at<uchar>(i, j)];
		
	imshow("output_line_gray", output_line_gray);

	//gamma矫正例子1
	float gamma = 1.3;
	uchar LUT_gamma[256];
	for (int i = 0; i < 256; ++i)
	{
		int result = pow(i / 255.0, gamma) * 255;
		if (result <= 255)
			LUT_gamma[i] = result;
		else
			LUT_gamma[i] = 255;
	}

	Mat output_gamma_gray;
	input_gray.copyTo(output_gamma_gray);
	for (int i = 0; i < h; ++i)
		for (int j = 0; j < w; ++j)
			output_gamma_gray.at<uchar>(i, j) = LUT_gamma[input_gray.at<uchar>(i, j)];
	imshow("output_gamma_gray", output_gamma_gray);

	//三通道直方图均衡
	vector<Mat> input_channels;
	split(input, input_channels);
	vector<Mat> output_channels;
	for (int chan = 0; chan < 3; ++chan)
	{
		Mat output_equalize_chan;
		equalizeHist(input_channels.at(chan), output_equalize_chan);
		output_channels.push_back(output_equalize_chan);
	}
	Mat output_equalize;
	merge(output_channels, output_equalize);
	imshow("output_equalize", output_equalize);

	//gamma矫正例子2
	Mat input_exam2 = imread("../../assets/gamma_example.jpg");
	h = input_exam2.rows;
	w = input_exam2.cols;
	imshow("input_exam2", input_exam2);
	float gamma_exam2 = 0.3;
	uchar LUT_gamma_exam2[256];
	for (int i = 0; i < 256; ++i)
	{
		int result = pow(i / 255.0, gamma_exam2) * 255;
		if (result <= 255)
			LUT_gamma_exam2[i] = result;
		else
			LUT_gamma_exam2[i] = 255;
	}

	Mat output_exam2_gamma;
	input_exam2.copyTo(output_exam2_gamma);
	for (int i = 0; i < h; ++i)
		for (int j = 0; j < w; ++j)
			for (int chan = 0; chan < 3; ++chan)
				output_exam2_gamma.at<Vec3b>(i, j)[chan] = LUT_gamma_exam2[input_exam2.at<Vec3b>(i, j)[chan]];
		
	imshow("output_exam2_gamma", output_exam2_gamma);

	waitKey(0);
}