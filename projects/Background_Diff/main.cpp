#include <opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

int drawHist(Mat& histMat, int histgram[], int bin_width, int bin_heght);
void on_mouse(int EVENT, int x, int y, int flags, void* userdata);
int verifyGaussian();
void threshold_track(int th, void* data);
int bgSub_demo();
int opencvBgSubtrator(int BG_METHOD);

String wName = "test";
Point vP;
int sub_threshold;

int drawHist(Mat& histMat, int histgram[], int bin_width, int bin_heght)
{
    int width = bin_width * 256;
    Mat hMat(bin_heght, width, CV_8UC3, Scalar(255, 255, 255));

    for (int i = 0; i < 256; i++)
    {
        rectangle(hMat, Point(bin_width * i, bin_heght - (float)histgram[i] * 1),
                  Point(bin_width * (i + 1), bin_heght),
                  Scalar(255, 0, 0), -1);
    }
    hMat.copyTo(histMat);
    return 0;
}

void on_mouse(int EVENT, int x, int y, int flags, void* userdata)
{
    Mat hh;
    hh = *(Mat*)userdata;
    switch (EVENT)
    {
    case EVENT_LBUTTONDOWN:
    {
        vP.x = x;
        vP.y = y;
        circle(hh, vP, 4, Scalar(255, 255, 255), -1);
        imshow(wName, hh);
    }
    break;
    }
}

//��demo��֤����ʾ����Ƶ�е����ػҶ�ֵ�任�Ƿ�ʸ�˹�ֲ�
int verifyGaussian()
{
    //----------------------��ȡ��Ƶ�ļ�--------------------------
    VideoCapture capVideo(0);

    //�����Ƶ��ʧ��
    if (!capVideo.isOpened()) {
        std::cout << "Unable to open video!" << std::endl;
        return -1;
    }

    int cnt = 0;
    int bin_width = 3;
    int bin_heght = 100;
    int histgram[256] = { 0 };

    cv::Mat histMat;

    while (1) {

        Mat frame;
        Mat grayMat;
        capVideo >> frame;

        if (frame.empty()) {
            std::cout << "Unable to read frame!" << std::endl;
            return -1;
        }

        //��һ֡ѡȡ����
        if (cnt == 0) {
            Mat selectMat;
            frame.copyTo(selectMat);
            namedWindow(wName);
            imshow(wName, selectMat);
            setMouseCallback(wName, on_mouse, &selectMat);
            waitKey(0);
            destroyAllWindows();
        }

        cvtColor(frame, grayMat, COLOR_BGR2GRAY);

        //������ػҶ�ֵ
        int index = grayMat.at<uchar>(vP.y, vP.x);
        //ֱ��ͼ��Ӧ��bin��1
        histgram[index]++;

        //����ֱ��ͼ
        drawHist(histMat, histgram, bin_width, bin_heght);

        drawMarker(frame, vP, Scalar(255, 255, 255));
        imshow("frame", frame);
        imshow("histMat", histMat);

        //��ʾͼƬ����ʱ30ms������Ҫ��waitKey()�������޷���ʾͼ��
        //�ȴ�������Ӧ������ESC���˳�
        if (waitKey(30) == 27) {
            destroyAllWindows();
            break;
        }
        cnt++;
    }
    return 0;
}

void threshold_track(int th, void* data)
{
    sub_threshold = th;
}

//�ֶ�ʵ�ֱ������
int bgSub_demo()
{
    //----------------------��ȡ��Ƶ�ļ�--------------------------
    VideoCapture capVideo(0);

    //�����Ƶ��ʧ��
    if (!capVideo.isOpened()) {
        std::cout << "Unable to open video!" << std::endl;
        return -1;
    }

    //������
    int cnt = 0;
    Mat frame, bgMat;
    while (1) {
        capVideo >> frame;
        cvtColor(frame, frame, COLOR_BGR2GRAY);

        if (frame.empty()) {
            std::cout << "Unable to read frame!" << std::endl;
            return -1;
        }

        if (cnt == 0) {
            //��һ֡����ñ���ͼ��
            frame.copyTo(bgMat);
            namedWindow("Result", WINDOW_AUTOSIZE);
            cv::createTrackbar("threshold", "Result", 0, 255, threshold_track);
            threshold_track(0, 0);
        }
        else {
            //�ڶ�֡��ʼ�������
            //����ͼ��͵�ǰͼ�����
            Mat subMat, thMat, thMat2;
            absdiff(frame, bgMat, subMat);
            threshold(subMat, thMat, sub_threshold, 255, THRESH_BINARY);

            imshow("frame", frame);
            imshow("thMat", thMat);


            threshold(subMat, thMat2, 40, 255, THRESH_BINARY);
            //imshow("thMat2", thMat2);

            //������
            Mat maskMat_close;
            morphologyEx(thMat2, maskMat_close, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(9, 9)));
            //imshow("maskMat_close", maskMat_close);

            //��ͨ����
            Mat labels, stats, centroids;
            int nComp = connectedComponentsWithStats(maskMat_close, labels, stats, centroids, 8, CV_32S);

            Mat output;
            frame.copyTo(output);
            //������
            int maxi = 0, maxS = 0;
            Rect maxbandbox;
            for (int i = 1; i < nComp; i++)
            {
                Rect bandbox;
                bandbox.x = stats.at<int>(i, 0);
                bandbox.y = stats.at<int>(i, 1);

                bandbox.width = stats.at<int>(i, 2);
                bandbox.height = stats.at<int>(i, 3);
                int S = bandbox.width * bandbox.height;
                if (S > maxS)
                {
                    maxi = i;
                    maxS = S;
                    maxbandbox = bandbox;
                }
                
                //rectangle(output, bandbox, Scalar(255, 0, 0), 1, 8, 0);
            }
            rectangle(output, maxbandbox, Scalar(255, 0, 0), 1, 8, 0);
            imshow("output", output);

        }

        //��ʾͼƬ����ʱ30ms������Ҫ��waitKey()�������޷���ʾͼ��
        //�ȴ�������Ӧ������ESC���˳�
        if (waitKey(30) == 27) {
            destroyAllWindows();
            break;
        }
        cnt++;
    }

    return 0;
}

//����opencv�ı�����ֺ�������
int opencvBgSubtrator(int BG_METHOD)
{
    //----------------------��ȡ��Ƶ�ļ�--------------------------
    VideoCapture capVideo(0);

    //�����Ƶ��ʧ��
    if (!capVideo.isOpened()) {
        std::cout << "Unable to open video!" << std::endl;
        return -1;
    }

    //�����Ƶ��ʧ��
    if (!capVideo.isOpened()) {
        std::cout << "Unable to open video!" << std::endl;
        return -1;
    }

    Mat inputFrame, frame, foregroundMask, foreground, background;

    int method = BG_METHOD;
    Ptr<BackgroundSubtractor> model;
    if (method == 0) {
        model = createBackgroundSubtractorKNN();
    }
    else if (method == 1) {
        model = createBackgroundSubtractorMOG2();
    }
    else {
        cout << "Can not create background model using provided method: '" << method << "'" << endl;
    }

    bool doUpdateModel = true;
    bool doSmoothMask = false;

    while (1) {
        capVideo >> frame;

        if (frame.empty()) {
            std::cout << "Unable to read frame!" << std::endl;
            return -1;
        }

        // pass the frame to background model
        model->apply(frame, foregroundMask, doUpdateModel ? -1 : 0);

        // show processed frame
        imshow("image", frame);

        // show foreground image and mask (with optional smoothing)
        if (doSmoothMask)
        {
            GaussianBlur(foregroundMask, foregroundMask, Size(11, 11), 3.5, 3.5);
            threshold(foregroundMask, foregroundMask, 10, 255, THRESH_BINARY);
        }
        if (foreground.empty())
            foreground.create(frame.size(), frame.type());
        foreground = Scalar::all(0);
        frame.copyTo(foreground, foregroundMask);
        imshow("foreground mask", foregroundMask);
        imshow("foreground image", foreground);

        // show background image
        model->getBackgroundImage(background);
        if (!background.empty())
            imshow("mean background image", background);

        // interact with user
        const char key = (char)waitKey(30);
        if (key == 27 || key == 'q') // ESC
        {
            cout << "Exit requested" << endl;
            break;
        }
        else if (key == ' ')
        {
            doUpdateModel = !doUpdateModel;
            cout << "Toggle background update: " << (doUpdateModel ? "ON" : "OFF") << endl;
        }
        else if (key == 's')
        {
            doSmoothMask = !doSmoothMask;
            cout << "Toggle foreground mask smoothing: " << (doSmoothMask ? "ON" : "OFF") << endl;
        }
    }
    return 0;
}

int main()
{
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);
    //verifyGaussian();
    bgSub_demo();
    //opencvBgSubtrator(0);
}