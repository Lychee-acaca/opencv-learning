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

	//反色
	srcMat = 255 - srcMat;

	//大津法二值化
	threshold(srcMat, thresh_Mat, 100, 255, THRESH_OTSU);

	//腐蚀
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
	erode(thresh_Mat, erode_Mat, element, Point(-1, -1), 2);

	//连通域检测
	int nComp = connectedComponentsWithStats(erode_Mat, labels, stats, centroids, 8, CV_32S);

	cout << "原点个数为：" << nComp - 1 << endl;

	imshow("srcMat", srcMat);
	imshow("thresh_Mat", thresh_Mat);
	imshow("erode_Mat", erode_Mat);
	waitKey(0);
}