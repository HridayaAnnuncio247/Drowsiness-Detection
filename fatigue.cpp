class fatigue
{
	public:
	
	utils ut;
	double thld;
	frontal_face_detector detector;
	shape_predictor sp;


	fatigue()
	{
		thld = 0.0;
		detector = None;
		sp = None;
	}

	void load_models
	{
		detector = get_frontal_face_detector();
		dlib::deserialize("C:/Users/HridayaAnnuncio/Documents/Face_Alignment/shape_predictor_68_face_landmarks.dat") >> sp;

	}

	full_object_detection get_landmarks(cv::Mat frame)
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

                int startX = x1 - 50, startY = y1 - 50, width = x2 - x1 +100, height = y2 - y1 +100;
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

        cap = ut.vid_capture(vid);
        int f=0;
        while (!win.is_closed())
        {
            cv::Mat frame;
            cv::Mat frame2;

            bool bSuccess = cap.read(frame); // read a new frame from video 

            if (!cap.read(frame))
            {
                break;
            }
            f=f+1;

            shape = get_landmarks(frame);

             		double leftEAR = ut.eye_aspect_ratio(lear);
                    double rightEAR = ut.eye_aspect_ratio(rear);
                    double mar = ut.mouth_aspect_ratio(mr);


                    double avg = (leftEAR + rightEAR) / 2;
                    auto nw = chrono::steady_clock::now();
                    //auto now = std::chrono::system_clock::now();
                    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(nw);

                    auto value = now_ms.time_since_epoch();
                    double tm = value.count();




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

			// Now let's view our face poses on the screen.
            win.clear_overlay();
            win.set_image(gimg2);
            win.add_overlay(render_face_detections(shapes));



        }
            
            
        auto now_ms1 = std::chrono::time_point_cast<std::chrono::milliseconds>(chrono::steady_clock::now());
        auto value1 = now_ms1.time_since_epoch();
        double et = value1.count();

        cout << "no. of frames = " << f << endl;
        cout << "time taken = " << et - st << endl;
        cout << "fps = " << f / (et - st) << endl;

        

     }

	


};


int main()
{
	try{
		fatigue ft;
		ft.detect fatigue("0");
	}
	catch(exception & e)
	{
		cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
	}
}