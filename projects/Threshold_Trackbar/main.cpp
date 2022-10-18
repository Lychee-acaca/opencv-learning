#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

//�ص�������һ������thΪ�϶�����λ�ã��ڶ�������Ϊ�����img
void onChange(int th, void* data)
{
	Mat src = *(Mat*)(data);
	Mat dst;

	//�Ҷȴ�����ֵΪ���ֵ������ֵΪ0
	threshold(src, dst, th, 255, THRESH_BINARY);

	//����Զ�Ѱ��ȫ����ֵ����Ϊ���Զ�������thû���ã���ô����������һ����
	//threshold(src, dst, th, 255, THRESH_OTSU);

	//����Ӧ��ֵ������������г��������������Ҳû����
	//adaptiveThreshold(src, dst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 15, 10);

	imshow("img", dst);
}

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);

	//����Ҷ�ͼ
	Mat img = imread("../../assets/grid.jpg", IMREAD_GRAYSCALE);

	//��ʼ��ֵ�趨Ϊ0
	Mat init;
	threshold(img, init, 0, 255, 0);
	imshow("img", init);

	/*����һ������
	* ��������ΪThresholdTrackbar
	* ������img����
	* ��ʼλ��0�����ֵ255
	* �ص�����ΪonChange�����������϶�����ûص�����
	* ��img�ĵ�ַ�����ص�����
	*/
	createTrackbar("ThresholdTrackbar", "img", 0, 255, onChange, &img);
	waitKey(0);
}