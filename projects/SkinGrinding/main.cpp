#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);

	//��ȡ��תHSV
	Mat hsvMat;
	Mat input = imread("../../assets/WangSiTu.jpg");
	float scale = 1.0;
	resize(input, input, Size(input.cols * scale, input.rows * scale));
	cvtColor(input, hsvMat, COLOR_BGR2HSV);


	//����Ƥ��HSV��Χ����ͼ���е�Ƥ������
	double i_minH = 0, i_maxH = 20;
	double i_minS = 43, i_maxS = 255;
	double i_minV = 55, i_maxV = 255;

	Mat maskMat;
	inRange(hsvMat, Scalar(i_minH, i_minS, i_minV), Scalar(i_maxH, i_maxS, i_maxV), maskMat);

	//�˲�
	Mat filterOut;

	//˫���˲�
	int bilateralFilterVal = 30;
	bilateralFilter(input, filterOut, bilateralFilterVal, bilateralFilterVal * 2, bilateralFilterVal / 2);
	
	//��˹�˲�
	//GaussianBlur(input, filterOut, Size(5, 5), 3, 0);

	//��ֵ�˲�
	//medianBlur(input, filterOut, 7);

	//��ֵ�˲�
	//blur(input, filterOut, Size(3, 3));

	//���˲�ͼ��Ƥ������Ĳ��ֿ��������ͼ��
	Mat objectMat;
	input.copyTo(objectMat);
	filterOut.copyTo(objectMat, maskMat);

	//��ʾ���
	imshow("ĥƤǰ", input);
	imshow("ĥƤ��", objectMat);
	waitKey(0);
}