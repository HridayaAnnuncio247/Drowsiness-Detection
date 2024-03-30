
#include <c:/dlib/image_processing/frontal_face_detector.h>
#include <c:/dlib/image_processing/render_face_detections.h>
#include <c:/dlib/image_processing.h>
#include <c:/dlib/gui_widgets.h>
#include <c:/dlib/image_io.h>
#include <c:/dlib/opencv/cv_image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include<math.h>
//#include<unistd.h>
#include<chrono>


#include<iostream>
//using namespace _IMAGE_POLICY_ENTRY::__unnamed_union_0469_172;
//using namespace _IMAGE_POLICY_ENTRY::__unnamed_union_0469_172;
using namespace std;
using namespace dlib;
using namespace std::chrono;

class utils
{
public:

    double euclidean(double x1, double y1, double x2, double y2)
    {
        double x = x1 - x2; //calculating number to square in next step
        double y = y1 - y2;
        double dist;

        dist = pow(x, 2) + pow(y, 2);       //calculating Euclidean distance
        dist = sqrt(dist);

        return dist;
    }

    double eye_aspect_ratio(std::vector<std::vector<int>> eye)
    {

        double A = euclidean(eye[1][0], eye[1][1], eye[5][0], eye[5][1]);
        double B = euclidean(eye[2][0], eye[2][1], eye[4][0], eye[4][1]);


        // compute the euclidean distance between the horizontal
        //# eye landmark (x, y)-coordinates
        double C = euclidean(eye[0][0], eye[0][1], eye[3][0], eye[3][1]);


        //# compute the eye aspect ratio
        double EAR = (A + B) / (2.0 * C);
        return EAR;
    }

    double mouth_aspect_ratio(std::vector<std::vector<int>> eye)
    {

        double A = euclidean(eye[0][0], eye[0][1], eye[1][0], eye[1][1]);
        double B = euclidean(eye[2][0], eye[2][1], eye[3][0], eye[3][1]);
        double C = euclidean(eye[4][0], eye[4][1], eye[5][0], eye[5][1]);



        // compute the euclidean distance between the horizontal
        //# mouth landmark (x, y)-coordinates
        double D = euclidean(eye[6][0], eye[6][1], eye[7][0], eye[7][1]);


        //# compute the eye aspect ratio
        double EAR = (A + B + C) / (3.0 * D);
        return EAR;
    }

    cv::VideoCapture vid_capture(string vid)
    {
        
        if (vid.compare("0") == 0)
        {
            cv::VideoCapture cap(0);
            return cap;

        }
        else
        {
            cv::VideoCapture cap(vid);
            return cap;

        }



    }



};


class fatigue
{
public:

    utils ut;
    double thld;
    dlib::frontal_face_detector detector;
    dlib::shape_predictor sp;


    fatigue()
    {
        ut = utils();
        thld = 0.0;
        //detector = _IMAGE_POLICY_ENTRY::__unnamed_union_0469_172::None;
        //sp = _IMAGE_POLICY_ENTRY::__unnamed_union_0469_172::None;
    }

    void load_models()
    {
        detector = dlib::get_frontal_face_detector();
        dlib::deserialize("C:/Users/HridayaAnnuncio/Documents/Face_Alignment/shape_predictor_68_face_landmarks.dat") >> sp;

    }

    dlib::full_object_detection get_landmarks(cv::Mat frame)
    {
        cv::Mat Gray;
        cvtColor(frame, Gray, cv::COLOR_BGR2GRAY);
        dlib::array2d<unsigned char> gimg;
        dlib::assign_image(gimg, dlib::cv_image<unsigned char>(Gray));




        std::vector<rectangle> faces = detector(gimg);

        for (unsigned long j = 0; j < faces.size(); ++j)
        {
            cv::Mat Rgb;
            cvtColor(Gray, Rgb, cv::COLOR_GRAY2BGR);


            int x1 = faces[j].left();
            int y1 = faces[j].top();
            int x2 = faces[j].right();
            int y2 = faces[j].bottom();

            int startX = x1 - 50, startY = y1 - 50, width = x2 - x1 + 100, height = y2 - y1 + 100;
            cv::Mat adjusted(Rgb, cv::Rect(startX, startY, width, height));



            std::vector<cv::Mat> adjs(3);
            cv::split(adjusted, adjs);

            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
            clahe->setClipLimit(4);
            //cv::Mat adjusted;
            clahe->apply(adjs[0], adjusted);

            adjusted.copyTo(adjs[0]);
            cv::merge(adjs, adjusted);

            cv::Mat adjusted_clahe;
            cv::cvtColor(adjusted, adjusted_clahe, cv::COLOR_Lab2BGR);

            cv::Mat Gray2;
            cvtColor(adjusted_clahe, Gray2, cv::COLOR_BGR2GRAY);


            //array2d<rgb_pixel> img;
            //dlib::assign_image(img,dlib::cv_image<dlib::bgr_pixel>(frame));
            dlib::array2d<unsigned char> gimg2;
            dlib::assign_image(gimg2, dlib::cv_image<unsigned char>(Gray2));



            std::vector<rectangle> faces2 = detector(gimg2);
            //cout << "Number of faces detected: " << dets.size() << endl;

            // Now we will go ask the shape_predictor to tell us the pose of
            // each face we detected.
            //std::vector<full_object_detection> shapes;
            for (unsigned long j = 0; j < faces2.size(); ++j)
            {
                full_object_detection shape = sp(gimg2, faces2[j]);
                return shape;
            }
        }
    }

    /*double get_threshold(string vid)
    {
        cap = ut.vid_capture(vid);

    }*/

    void detect_fatigue(string vid)
    {
        double min_avg = 0.0;
        double min_tm = 0.0;
        double max_tm = 0.0;

        double min_avg_m = 0.0;
        double min_tm_m = 0.0;
        double max_tm_m = 0.0;

        image_window win, win_faces;
        cv::String window_name = "Video";
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);

        cv::VideoCapture cap = ut.vid_capture(vid);
        cout << "detect fatigue video captured" << endl;
        int f = 0;
        while (!win.is_closed())
        {
            cout << "inside while loop" << endl;
            cv::Mat frame;
            cv::Mat frame2;

            bool bSuccess = cap.read(frame); // read a new frame from video 
            cout << "frame read" << endl;

            if (!cap.read(frame))
            {
                cout << "Can't read frame!!" << endl;
                break;
            }
            f = f + 1;

            full_object_detection shape = get_landmarks(frame);
            cout << "landmarks read" << endl;

            std::vector<std::vector<int>> lear{ {shape.part(36)(0),shape.part(36)(1)},{shape.part(37)(0),shape.part(37)(1)},{shape.part(38)(0),shape.part(38)(1)}, {shape.part(39)(0),shape.part(39)(1)},{shape.part(40)(0),shape.part(40)(1)},{shape.part(41)(0),shape.part(41)(1)} };
 
            std::vector<std::vector<int>> rear{ {shape.part(42)(0),shape.part(42)(1)},{shape.part(43)(0),shape.part(43)(1)},{shape.part(44)(0),shape.part(44)(1)}, {shape.part(45)(0),shape.part(45)(1)},{shape.part(46)(0),shape.part(46)(1)},{shape.part(47)(0),shape.part(47)(1)} };

            std::vector<std::vector<int>> mr{ {shape.part(61)(0),shape.part(61)(1)},{shape.part(67)(0),shape.part(67)(1)},{shape.part(62)(0),shape.part(62)(1)}, {shape.part(66)(0),shape.part(66)(1)},{shape.part(63)(0),shape.part(63)(1)},{shape.part(65)(0),shape.part(65)(1)},{shape.part(60)(0),shape.part(60)(1)},{shape.part(64)(0),shape.part(64)(1)} };
            cout << "vectors initialized" << endl;

            double leftEAR = ut.eye_aspect_ratio(lear);
            double rightEAR = ut.eye_aspect_ratio(rear);
            double mar = ut.mouth_aspect_ratio(mr);

            cout << "aspect ratios done" << endl;


            double avg = (leftEAR + rightEAR) / 2;
            auto nw = chrono::steady_clock::now();
            //auto now = std::chrono::system_clock::now();
            auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(nw);

            auto value = now_ms.time_since_epoch();
            double tm = value.count();

            cout << "done all initializations" << endl;


            if (min_avg == 0.0)
            {
                min_avg = avg;
                min_tm = tm;
                //min_tm = dur.count();
            }
            else
            {
                if (avg<(min_avg + 0.03) && avg>(min_avg - 0.03) && avg < 0.15)
                {
                    max_tm = tm;
                    if (max_tm - min_tm >= 1000)
                    {
                        cout << max_tm - min_tm << endl;
                        cout << "Drowsy Behaviour";
                        min_avg = 0.0;
                    }
                }
                else
                {
                    min_avg = avg;
                    min_tm = tm;
                }
            }

            cout << "ended fatigue detection if elses" << endl;

            if (min_avg_m == 0.0)
            {
                min_avg_m = mar;
                min_tm_m = tm;
            }
            else
            {
                if (mar<(min_avg_m + 0.05) && mar>(min_avg_m - 0.05) && mar > 0.30)
                {
                    max_tm_m = tm;
                    if ((max_tm_m - min_tm_m) >= 1000)
                    {
                        cout << (max_tm_m - min_tm_m) << endl;
                        cout << "Yawning!!" << endl;
                        min_avg_m = 0.0;

                    }
                }

                else
                {
                    min_avg_m = mar;
                    min_tm_m = tm;
                }

            }
            cout << "completed yawn detection if elses" << endl;
            // Now let's view our face poses on the screen.
            //win.clear_overlay();
            //win.set_image(gimg2);
            //win.add_overlay(render_face_detections(shapes));

            cout << "no. of frames = " << f << endl;


        }






    }




};



int main()

{
    try {
        cout << "entered main" << endl;

        fatigue ft;

        auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(chrono::steady_clock::now());
        auto value = now_ms.time_since_epoch();
        double st = value.count();

        ft.detect_fatigue("0");

        cout << "fatigue detected" << endl;

        auto now_ms1 = std::chrono::time_point_cast<std::chrono::milliseconds>(chrono::steady_clock::now());
        auto value1 = now_ms1.time_since_epoch();
        double et = value1.count();

        cout << "time taken = " << et - st << endl;
    }
    catch (exception & e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }


}
