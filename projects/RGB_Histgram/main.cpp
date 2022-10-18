#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);

	//调用本机摄像头，统计每个像素点的BGR值，用line函数绘制直方图
	//实现的比较糙，也不想再改了，暂时就这样吧
	VideoCapture cap(0);
	while (1)
	{
		int histgramSum[256][3];
		float histgram[256][3];
		Mat frame;
		cap >> frame;

		int h = frame.rows;
		int w = frame.cols;

		memset(histgramSum, 0, sizeof(histgramSum));

		for (int ch = 0; ch < 3; ch++)
			for (int j = 0; j < h; ++j)
				for (int i = 0; i < w; ++i)
					histgramSum[frame.at<Vec3b>(j, i)[ch]][ch]++;

		for (int ch = 0; ch < 3; ch++)
			for (int i = 0; i < 256; ++i)
				histgram[i][ch] = histgramSum[i][ch] / (float)(h * w);

		Mat output = Mat::zeros(300, 256, CV_8UC3);

		for (int i = 0; i < 256; ++i)
		{
			line(output, Point(i, 100), Point(i, 100 - histgram[i][0] * 3000), Scalar(0, 0, 255), 3, LINE_4);
			line(output, Point(i, 200), Point(i, 200 - histgram[i][1] * 3000), Scalar(0, 255, 0), 3, LINE_4);
			line(output, Point(i, 300), Point(i, 300 - histgram[i][2] * 3000), Scalar(255, 0, 0), 3, LINE_4);
		}

		const float scale = 2;
		resize(frame, frame, Size(frame.cols * scale, frame.rows * scale));
		resize(output, output, Size(output.cols * scale, output.rows * scale));

		imshow("input", frame);
		imshow("output", output);
		waitKey(30);
	}
	return 0;
}

