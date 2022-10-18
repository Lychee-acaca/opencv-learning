#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);
	Mat srcMat = imread("../../assets/SolderJoint.JPG", IMREAD_GRAYSCALE);

	Mat thresh_Mat, erode_Mat;
	Mat labels, stats, centroids;

	//��ɫ
	srcMat = 255 - srcMat;

	//��򷨶�ֵ��
	threshold(srcMat, thresh_Mat, 100, 255, THRESH_OTSU);

	//��ʴ
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
	erode(thresh_Mat, erode_Mat, element, Point(-1, -1), 2);

	//��ͨ����
	int nComp = connectedComponentsWithStats(erode_Mat, labels, stats, centroids, 8, CV_32S);

	cout << "ԭ�����Ϊ��" << nComp - 1 << endl;

	imshow("srcMat", srcMat);
	imshow("thresh_Mat", thresh_Mat);
	imshow("erode_Mat", erode_Mat);
	waitKey(0);
}