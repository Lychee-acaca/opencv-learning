#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);

	Mat input = imread("../../assets/chip.png");
	imshow("input", input);

	//�Ҷ�
	Mat input_gray;
	cvtColor(input, input_gray, COLOR_BGR2GRAY);
	imshow("input_gray", input_gray);

	//��ֵ��
	Mat thresh_Mat;
	threshold(input_gray, thresh_Mat, 100, 255, THRESH_OTSU);
	imshow("thresh_Mat", thresh_Mat);

	//��ֵ�˲�
	Mat blur_Mat;
	medianBlur(thresh_Mat, blur_Mat, 9);
	imshow("blur_Mat", blur_Mat);

	//��ͨ����
	Mat labels, stats, centroids;
	int nComp = connectedComponentsWithStats(blur_Mat, labels, stats, centroids, 8, CV_32S);

	//ɸѡ����
	Mat output;
	input.copyTo(output);
	for (int i = 1; i < nComp; i++)
	{
		int width = stats.at<int>(i, 2);
		int height = stats.at<int>(i, 3);
		float p = (float)width / height;

		//��߱�����������������
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