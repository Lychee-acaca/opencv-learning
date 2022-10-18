//Ӧ��ֻ�����ڽ�����Եû�к�ɫ���ص�ͼƬ
//������Щ̫��ֵĽǶ�Ҳ��̫��
#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);

	Mat input = imread("../../assets/WangSiTu.jpg");

	//�Ȱ�ͼƬ������תһ���Ƕȣ���Ϊ�����㷨������
	//Ϊ�˷�ֹ��ʧ��ת��ı߽���Ϣ����ͼ���������
	float angel = 35.5;
	float h = sqrt(2) * cos(CV_PI / 4 - angel * 2 * CV_PI / 360);
	float scale = 1 / h;

	Point2f center(input.cols * 0.5, input.rows * 0.5);
	Mat affine_matrix = getRotationMatrix2D(center, angel, scale);
	Mat input_Rotated;
	warpAffine(input, input_Rotated, affine_matrix, input.size());

	//ͼ���������
	Mat maskMat;
	//��ȡͼƬ�ĺ�ɫ���򣬽��б�Ե��ȡ
	inRange(input_Rotated, Scalar(0, 0, 0), Scalar(0, 0, 0), maskMat);
	imshow("step1", maskMat);
	Canny(maskMat, maskMat, 1, 10);
	imshow("step2", maskMat);

	//���������ն�����
	//�����б����㣬��ʹ��canny��Ե��⣬���ܻᶪʧ�ĸ���
	morphologyEx(maskMat, maskMat, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(7, 7)));
	imshow("step3", maskMat);

	vector<vector<Point>> contours;
	vector<Vec4i> hierachy;

	//��ȡ����
	findContours(maskMat, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(-1, -1));

	//�����������ı��
	int index = -1;
	int max = 0;
	for (size_t i = 0; i < contours.size(); i++)
	{
		//��ȡ�������
		double area = contourArea(contours[i]);
		if (area > max)
		{
			max = area;
			index = i;
		}
	}

	//�����������С��Ӿ���
	RotatedRect rect = minAreaRect(contours[index]);
	
	//����ε��ĸ���
	Point2f rectp[4];
	rect.points(rectp);

	//���廭�����ĸ�����
	Point2f pts[] = {
		Point2f(0, 0),
		Point2f(input_Rotated.rows, 0),
		Point2f(input_Rotated.rows, input_Rotated.cols),
		Point2f(0, input_Rotated.cols),
	};

	//����ͶӰ�任
	Mat output;
	Mat perspective_matrix = getPerspectiveTransform(rectp, pts);
	warpPerspective(input_Rotated, output, perspective_matrix, output.size());

	imshow("input", input);
	imshow("����ǰ", input_Rotated);
	imshow("������", output);
	waitKey(0);
}
