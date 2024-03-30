
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



//using namespace cv;
using namespace std;
using namespace dlib;
using namespace std::chrono;
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

int main()
{
    try
    {
        int f = 0;//, st, et;
        int fs, fs_m, fe, fe_m;

        // This example takes in a shape model file and then a list of images to
        // process.  We will take these filenames in as command line arguments.
        // Dlib comes with example images in the examples/faces folder so give
        // those as arguments to this program.
        /*if (argc == 1)
        {
            cout << "Call this program like this:" << endl;
            cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
            cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
            cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
            return 0;
        }*/

        //auto nw = chrono::steady_clock::now();
        //auto now = std::chrono::system_clock::now();
        auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(chrono::steady_clock::now());
        auto value = now_ms.time_since_epoch();
        double st = value.count();

        cv::VideoCapture cap("C:/Users/HridayaAnnuncio/Documents/Face_Alignment/yawn_ashish.mp4");

        //int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH)); //get the width of frames of the video
        //int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

        int frame_width = static_cast<int>(cap.get(3)); //get the width of frames of the video
        int frame_height = static_cast<int>(cap.get(4)); //get the height of frames of the video

        cv::Size frame_size(frame_width, frame_height);
        int frames_per_second = 30;

        //Create and initialize the VideoWriter object 
        cv::VideoWriter oVideoWriter("C:/Users/HridayaAnnuncio/Documents/Face_Alignment/yawn_ashish_edited1.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
            frames_per_second, frame_size, false);

        // We need a face detector.  We will use this to get bounding boxes for
        // each face in an image.

        frontal_face_detector detector = get_frontal_face_detector();

        // And we also need a shape_predictor.  This is the tool that will predict face
        // landmark positions given an image and face bounding box.  Here we are just
        // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
        // as a command line argument.

        shape_predictor sp;
        //int f=0,st,et;
        dlib::deserialize("C:/Users/HridayaAnnuncio/Documents/Face_Alignment/shape_predictor_68_face_landmarks.dat") >> sp;


        image_window win, win_faces;
        cv::String window_name = "Video";
        cv::namedWindow(window_name, cv::WINDOW_NORMAL); //create a window



        cv::Mat frame;
        dlib::array2d<unsigned char> gimg2;
        std::vector<full_object_detection> shapes;




        //std::vector<auto> time_ms;
        double min_avg = 0.0;
        double min_tm = 0.0;
        double max_tm = 0.0;

        double min_avg_m = 0.0;
        double min_tm_m = 0.0;
        double max_tm_m = 0.0;
        //auto tm = chrono::steady_clock::now();



        while (!win.is_closed())
        {
            cv::Mat frame;
            cv::Mat frame2;

            bool bSuccess = cap.read(frame); // read a new frame from video 

            if (!cap.read(frame))
            {
                break;
            }

            f = f + 1;
            //cv::Mat Gray;
            //cvtColor(frame, Gray, cv::COLOR_BGR2GRAY);

            /*cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
            clahe->setClipLimit(4);
            cv::Mat adjusted;
            clahe->apply(Gray, adjusted);*/


            //Breaking the while loop at the end of the video
            if (bSuccess == false)
            {
                cout << "Found the end of the video" << endl;
                break;
            }

            //show the frame in the created window
            imshow(window_name, frame);
            cv::Mat Gray;

            cvtColor(frame, Gray, cv::COLOR_BGR2GRAY);
            dlib::array2d<unsigned char> gimg;
            dlib::assign_image(gimg, dlib::cv_image<unsigned char>(Gray));




            std::vector<rectangle> faces = detector(gimg);
            for (unsigned long j = 0; j < faces.size(); ++j)
            {
                auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(chrono::steady_clock::now());
                auto value = now_ms.time_since_epoch();
                double st = value.count();
                //cv::Mat Rgb;
                //cvtColor(Gray, Rgb, cv::COLOR_GRAY2BGR);


                int x1 = faces[j].left();
                int y1 = faces[j].top();
                int x2 = faces[j].right();
                int y2 = faces[j].bottom();

                //int startX = x1 - 50, startY = y1 - 50, width = x2 - x1 + 100, height = y2 - y1 + 100;
                //cv::Mat adjusted(Rgb, cv::Rect(startX, startY, width, height));



                //std::vector<cv::Mat> adjs(3);
                //cv::split(Rgb, adjs);//adjusted, adjs);

                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->setClipLimit(2.5);
                //cv::Size s = (20, 20);
                clahe->setTilesGridSize(cv::Size(20, 20));
                cv::Mat Gray2;// adjusted;
                clahe->apply(Gray, Gray2);//adjs[0], Rgb);// adjusted);

                //Rgb.copyTo(adjs[0]);//adjusted.copyTo(adjs[0]);
                //cv::merge(adjs, Rgb);// adjusted);

                //cv::Mat adjusted_clahe;
                //cv::cvtColor(Rgb, adjusted_clahe, cv::COLOR_Lab2BGR);//adjusted, adjusted_clahe, cv::COLOR_Lab2BGR);

                //cv::Mat Gray2;
                //cvtColor(adjusted_clahe, Gray2, cv::COLOR_BGR2GRAY);


                //array2d<rgb_pixel> img;
                //dlib::assign_image(img,dlib::cv_image<dlib::bgr_pixel>(frame));
                //dlib::array2d<unsigned char> gimg2;
                dlib::assign_image(gimg2, dlib::cv_image<unsigned char>(Gray2));

                oVideoWriter.write(Gray2);

                std::vector<rectangle> faces2 = detector(gimg2);
                //cout << "Number of faces detected: " << dets.size() << endl;

                // Now we will go ask the shape_predictor to tell us the pose of
                // each face we detected.

                std::vector<full_object_detection> shapes;
                for (unsigned long j = 0; j < faces2.size(); ++j)
                {
                    full_object_detection shape = sp(gimg2, faces2[j]);

                    std::vector<std::vector<int>> lear{ {shape.part(36)(0),shape.part(36)(1)},{shape.part(37)(0),shape.part(37)(1)},{shape.part(38)(0),shape.part(38)(1)}, {shape.part(39)(0),shape.part(39)(1)},{shape.part(40)(0),shape.part(40)(1)},{shape.part(41)(0),shape.part(41)(1)} };
                    std::vector<std::vector<int>> rear{ {shape.part(42)(0),shape.part(42)(1)},{shape.part(43)(0),shape.part(43)(1)},{shape.part(44)(0),shape.part(44)(1)}, {shape.part(45)(0),shape.part(45)(1)},{shape.part(46)(0),shape.part(46)(1)},{shape.part(47)(0),shape.part(47)(1)} };

                    std::vector<std::vector<int>> mr{ {shape.part(61)(0),shape.part(61)(1)},{shape.part(67)(0),shape.part(67)(1)},{shape.part(62)(0),shape.part(62)(1)}, {shape.part(66)(0),shape.part(66)(1)},{shape.part(63)(0),shape.part(63)(1)},{shape.part(65)(0),shape.part(65)(1)},{shape.part(60)(0),shape.part(60)(1)},{shape.part(64)(0),shape.part(64)(1)} };



                    double leftEAR = eye_aspect_ratio(lear);
                    double rightEAR = eye_aspect_ratio(rear);
                    double mar = mouth_aspect_ratio(mr);


                    double avg = (leftEAR + rightEAR) / 2;
                    auto nw = chrono::steady_clock::now();
                    //auto now = std::chrono::system_clock::now();
                    auto now_ms = std::chrono::time_point_cast<std::chrono::seconds>(nw);

                    auto value = now_ms.time_since_epoch();
                    double tm = value.count();




                    if (min_avg == 0.0)
                    {
                        min_avg = avg;
                        min_tm = tm;
                        fs = f;
                        //min_tm = dur.count();
                    }
                    else
                    {
                        if (avg<(min_avg + 0.03) && avg>(min_avg - 0.03) && avg < 0.10)
                        {
                            max_tm = tm;
                            fe = f;
                            if (fe-fs>=6)//max_tm - min_tm >= 1000)
                            {
                                //cout << max_tm - min_tm << endl;
                                cout << "Drowsy Behaviour" << endl;
                                //cout << "Duration:" << min_tm  << " - " << max_tm  << endl;
                                cout << "frames:" << fs << "-" << fe << endl;
                                cout << endl;

                                min_avg = 0.0;
                            }
                        }
                        else
                        {
                            min_avg = avg;
                            min_tm = tm;
                            fs = f;
                        }
                    }

                    if (min_avg_m == 0.0)
                    {
                        min_avg_m = mar;
                        min_tm_m = tm;
                        fs_m = f;
                    }
                    else
                    {
                        if (mar<(min_avg_m + 0.05) && mar>(min_avg_m - 0.05) && mar > 0.30)
                        {
                            max_tm_m = tm;
                            fe_m = f;
                            if ((max_tm_m - min_tm_m) >= 1000)
                            {
                                //cout << (max_tm_m - min_tm_m) << endl;
                                cout << "Yawning!!" << endl;
                                min_avg_m = 0.0;
                                //cout << "Duration:" << min_tm_m  << " - " << max_tm_m  << endl;
                                cout << "frames:" << fs_m << "-" << fe_m << endl;
                                cout << endl;


                            }
                        }

                        else
                        {
                            min_avg_m = mar;
                            min_tm_m = tm;
                            fs_m = f;
                        }

                    }

                    //cout << "EAR=" << e_a_r;
                    /*cout << "number of parts: " << shape.num_parts() << endl;
                    cout << "pixel position of first part:  " << shape.part(0)(0) << endl;
                    cout << "pixel position of second part: " << shape.part(1)(1) << endl;
                    // You get the idea, you can get all the face part locations if*/

                    // you want them.  Here we just store them in shapes so we can
                    // put them on the screen.
                    shapes.push_back(shape);

                }
                win.clear_overlay();
                win.set_image(gimg2);
                win.add_overlay(render_face_detections(shapes));
            }










            // Now let's view our face poses on the screen.
            /*win.clear_overlay();
            win.set_image(frame);
            win.add_overlay(render_face_detections(shapes));*/

            // We can also extract copies of each face that are cropped, rotated upright,
            // and scaled to a standard size as shown here:
            //dlib::array<array2d<rgb_pixel> > face_chips;
            //extract_image_chips(adjusted, get_face_chip_details(shapes), face_chips);
            //win_faces.set_image(tile_images(face_chips));

            //cout << "Hit enter to process the next image..." << endl;
            //cin.get();
        }
        auto now_ms1 = std::chrono::time_point_cast<std::chrono::milliseconds>(chrono::steady_clock::now());
        auto value1 = now_ms1.time_since_epoch();
        double et = value1.count();

        cout << "no. of frames = " << f << endl;
        cout << "time taken = " << et - st << endl;
        cout << "fps = " << f / (et - st) << endl;



    }
    catch (exception & e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

