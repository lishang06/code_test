#include "shoulderDetect.h"

#include "fstream"
#include "iostream"
#include "string"
using namespace std;

float iouShoulder(shoulderRect box0, shoulderRect box1) 
{
    float xmin0 = box0.x;
    float ymin0 = box0.y;
    float xmax0 = box0.x + box0.width;
    float ymax0 = box0.y + box0.height;
    
    float xmin1 = box1.x;
    float ymin1 = box1.y;
    float xmax1 = box1.x + box1.width;
    float ymax1 = box1.y + box1.height;

    float w = fmax(0.0f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1));
    float h = fmax(0.0f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1));
    
    float i = w * h;
    float u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i;
    
    if (u <= 0.0) return 0.0f;
    else          return i/u;
}

float iouShoulder1(cv::Rect box0, cv::Rect box1) 
{
    float xmin0 = box0.x;
    float ymin0 = box0.y;
    float xmax0 = box0.x + box0.width;
    float ymax0 = box0.y + box0.height;
    
    float xmin1 = box1.x;
    float ymin1 = box1.y;
    float xmax1 = box1.x + box1.width;
    float ymax1 = box1.y + box1.height;

    float w = fmax(0.0f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1));
    float h = fmax(0.0f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1));
    
    float i = w * h;
    float u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i;
    
    if (u <= 0.0) return 0.0f;
    else          return i/u;
}

Inference_engineSD::Inference_engineSD()
{ }

Inference_engineSD::~Inference_engineSD()
{ 
    if ( netPtr_shoulderDetect != NULL )
	{
		if ( sessionPtr_shoulderDetect != NULL)
		{
			netPtr_shoulderDetect->releaseSession(sessionPtr_shoulderDetect);
			sessionPtr_shoulderDetect = NULL;
		}

		delete netPtr_shoulderDetect;
		netPtr_shoulderDetect = NULL;
	}
}

int Inference_engineSD::load_param(std::vector<std::string>& file, int num_thread)
{
    if (!file.empty())
    {
        if (file[0].find(".mnn") != std::string::npos)
        {
	        netPtr_shoulderDetect = MNN::Interpreter::createFromFile(file[0].c_str());
            if (nullptr == netPtr_shoulderDetect) 
            {
                while(1)
                {

                }
                return -1;
            }
			

            MNN::ScheduleConfig sch_config;
            sch_config.type = (MNNForwardType)MNN_FORWARD_CPU;

            MNN::BackendConfig backendConfig;
            //backendConfig.precision = MNN::BackendConfig::Precision_Normal;
            backendConfig.precision = MNN::BackendConfig::Precision_Low;
            sch_config.backendConfig = &backendConfig;

            num_thread = 4;
            if ( num_thread > 0 )sch_config.numThread = num_thread;
            sessionPtr_shoulderDetect = netPtr_shoulderDetect->createSession(sch_config);
            if (nullptr == sessionPtr_shoulderDetect) 
            {
                while(1)
                {
                    
                }
                return -1;
            }
        }
        else
        {
            while(1)
            {
                    
            }
            return -1;
        }
    }

    return 0;
}

/* 输出320*256上坐标或者320）128上坐标 */
int Inference_engineSD::shoulderDetect_infer_img(const char* path, cv::Mat& img, std::vector<cv::Rect> &shoulders, int flag)
{
    int input_width = 320;
    int input_height = 256;
    if( flag==2 )
    {
        input_width = 160;
        input_height = 128;
    }

    MNN::Tensor* tensorPtr = netPtr_shoulderDetect->getSessionInput(sessionPtr_shoulderDetect, nullptr);

    cv::Mat image;
    cv::resize(img, image, cv::Size(input_width, input_height));
    netPtr_shoulderDetect->resizeTensor(tensorPtr, {1, 3, input_height, input_width});
    netPtr_shoulderDetect->resizeSession(sessionPtr_shoulderDetect);
    float mean_vals[3] = {0.0f, 0.0f, 0.0f};
    float std_vals[3] = {1/255.0f, 1/255.0f, 1/255.0f};
    std::shared_ptr<MNN::CV::ImageProcess>
        pretreat(MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::BGR, mean_vals, 3, std_vals, 3));
    pretreat->convert(image.data, input_width, input_height, image.step[0], tensorPtr);
    netPtr_shoulderDetect->runSession(sessionPtr_shoulderDetect);

    std::string output1_str = "layer115-conv";
    std::string output2_str = "layer125-conv";

    MNN::Tensor *tensor_output1 = netPtr_shoulderDetect->getSessionOutput(sessionPtr_shoulderDetect, output1_str.c_str());
    MNN::Tensor *tensor_output2 = netPtr_shoulderDetect->getSessionOutput(sessionPtr_shoulderDetect, output2_str.c_str());

    MNN::Tensor tensor_output1_host(tensor_output1, tensor_output1->getDimensionType());
    MNN::Tensor tensor_output2_host(tensor_output2, tensor_output2->getDimensionType());

    tensor_output1->copyToHostTensor(&tensor_output1_host);
    tensor_output2->copyToHostTensor(&tensor_output2_host);

    auto output1_ptr = tensor_output1_host.host<float>();
    auto output2_ptr = tensor_output2_host.host<float>();

    int num_anchor1=3, num_anchor2=3;
    int num_class = 1;
    
    int grid_size1[2], grid_size2[2];
    int anchors1[6] , anchors2[6] ;
    if( flag==1 )
    {
        grid_size1[0] = 10; grid_size1[1] = 8;
        grid_size2[0] = 20; grid_size2[1] = 16;
        
        anchors1[0] = 45;   anchors1[0] = 58;
        anchors1[0] = 65;   anchors1[0] = 73;
        anchors1[0] = 88;   anchors1[0] = 106;
        
        anchors2[0] = 14;   anchors2[0] = 15;
        anchors2[0] = 22;   anchors2[0] = 28;
        anchors2[0] = 35;   anchors2[0] = 39;
    }
    else if( flag==2 )
    {
        grid_size1[0] = 5; grid_size1[1] = 4;
        grid_size2[0] = 10; grid_size2[1] = 8;
        
        anchors1[0] = 23;   anchors1[0] = 29;
        anchors1[0] = 33;   anchors1[0] = 37;
        anchors1[0] = 44;   anchors1[0] = 53;
        
        anchors2[0] = 7;   anchors2[0] = 8;
        anchors2[0] = 11;   anchors2[0] = 14;
        anchors2[0] = 18;   anchors2[0] = 20;
    }
    
    const int output_channel = 4 + num_class + 1; // 4 coord information and num of classes and 1 score information

    int gridnum1 = grid_size1[0] * grid_size1[1];
    int gridnum2 = grid_size2[0] * grid_size2[1];

    std::vector<shoulderRect> bbox_collection;

    float scoreThreshold = 0.6; //0.7 if you want more accurate result

    int k = 0;

    for (int i = 0; i < grid_size1[1];i++) //h
    {
        for (int j = 0; j < grid_size1[0];j++) //w
        {
            for (int m = 0; m < num_anchor1 * output_channel;m++)
            {
                if (m % output_channel == 4)
                {
                    k = m / output_channel;
                    float score = mnnSD_sigmoid(output1_ptr[m* gridnum1 + i * grid_size1[0] + j]);
                    if (score>scoreThreshold)
                    {
                        shoulderRect tmp_bbox;
                        int center_x = static_cast<int>((j+mnnSD_sigmoid(output1_ptr[(m-4)*gridnum1 + i*grid_size1[0]+j])) / grid_size1[0] * input_width);
                        int center_y = static_cast<int>((i+mnnSD_sigmoid(output1_ptr[(m-3)*gridnum1 + i*grid_size1[0]+j])) / grid_size1[1] * input_height);
                        tmp_bbox.width = static_cast<int>(exp(output1_ptr[(m-2)*gridnum1+i*grid_size1[0]+j])*anchors1[2*k]);
                        tmp_bbox.height = static_cast<int>(exp(output1_ptr[(m-1)*gridnum1+i*grid_size1[0]+j])*anchors1[2*k+1]);

                        //左上角点坐标
                        tmp_bbox.x = center_x - 0.5 * (float)tmp_bbox.width;
                        tmp_bbox.y = center_y - 0.5 * (float)tmp_bbox.height;
                        tmp_bbox.score = score;

                        bbox_collection.push_back(tmp_bbox);

                    }
                }
            }
        }
    }
    k = 0;
    for (int i = 0; i < grid_size2[1]; i++) //h
    {
        for (int j = 0; j < grid_size2[0];j++) //w
        {
            for (int m = 0; m < num_anchor2 * output_channel;m++)
            {
                if (m % output_channel == 4)
                {
                    k = m / output_channel;
                    float score = mnnSD_sigmoid(output2_ptr[m* gridnum2 + i * grid_size2[0] + j]);
                    if (score>scoreThreshold)
                    {
                        shoulderRect tmp_bbox;
                        int center_x = static_cast<int>((j+mnnSD_sigmoid(output2_ptr[(m-4)*gridnum2 + i*grid_size2[0]+j])) / grid_size2[0] * input_width);
                        int center_y = static_cast<int>((i+mnnSD_sigmoid(output2_ptr[(m-3)*gridnum2 + i*grid_size2[0]+j])) / grid_size2[1] * input_height);
                        tmp_bbox.width = static_cast<int>(exp(output2_ptr[(m-2)*gridnum2+i*grid_size2[0]+j])*anchors2[2*k]);
                        tmp_bbox.height = static_cast<int>(exp(output2_ptr[(m-1)*gridnum2+i*grid_size2[0]+j])*anchors2[2*k+1]);

                        //左上角点坐标
                        tmp_bbox.x = center_x - 0.5 * tmp_bbox.width;
                        tmp_bbox.y = center_y - 0.5 * tmp_bbox.height;
                        tmp_bbox.score = score;

                        bbox_collection.push_back(tmp_bbox);

                    }
                }
            }
        }
    }

    std::sort(bbox_collection.begin(), bbox_collection.end(), [](const shoulderRect a, const shoulderRect b) { return a.score > b.score; });
    std::vector<cv::Rect> bbox_result;
    int N = bbox_collection.size();
    std::vector<int> labels(N, -1);
    float nms_thresh = 0.45;
    for (int i = 0; i < N - 1; i++)
    {
        for (int j = i + 1; j < N;j++)
        {
            shoulderRect pre_box = bbox_collection.at(i);
            shoulderRect cur_box = bbox_collection.at(j);
            float iou_score = iouShoulder(pre_box, cur_box);
            if (iou_score > nms_thresh)
            {
                labels.at(j) = 0;
            }
        }
    }
    for (int i = 0; i < N;i++)
    {
        if (labels.at(i) == -1)
        {
            cv::Rect tmp_box;
            tmp_box.x = bbox_collection.at(i).x;
            tmp_box.y = bbox_collection.at(i).y;
            tmp_box.width = bbox_collection.at(i).width;
            tmp_box.height = bbox_collection.at(i).height;
            // tmp_box.score = bbox_collection.at(i).score;
            shoulders.push_back(tmp_box);
        }
    }
    
    return 0;

}



//int Inference_engineSD::shoulderDetect_infer_img(const char* path, cv::Mat& img, std::vector<cv::Rect> &shoulders, int flag)
//{
//    MNN::Tensor* tensorPtr = netPtr_shoulderDetect->getSessionInput(sessionPtr_shoulderDetect, nullptr);
//
//    int in_h = 320;
//    int in_w = 320;
//    if( flag==2 )
//    {
//        in_h = 160;
//        in_w = 160;
//    }
//
//    float mean_vals[3] = {0.0, 0.0, 0.0};
//    float norm_vals[3] = {1.0/255.0, 1.0/255.0, 1.0/255.0};
//    netPtr_shoulderDetect->resizeTensor(tensorPtr, {1, 3, in_h, in_w});
//    netPtr_shoulderDetect->resizeSession(sessionPtr_shoulderDetect);
//
//    std::shared_ptr<MNN::CV::ImageProcess> pretreat(
//            MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::BGR, mean_vals, 3,
//                                          norm_vals, 3));
//    pretreat->convert(img.data, in_w, in_h, img.step[0], tensorPtr);
//    auto tensorPtr_tensor = tensorPtr->host<float>();
//
//	netPtr_shoulderDetect->runSession(sessionPtr_shoulderDetect);
//
//    char* score_layer_name1 ="layer115-conv";  //layer79-conv 10*10
//    MNN::Tensor* tensor_scores1 = netPtr_shoulderDetect->getSessionOutput(sessionPtr_shoulderDetect, score_layer_name1);
//    MNN::Tensor tensor_scores_host1(tensor_scores1, tensor_scores1->getDimensionType());
//    tensor_scores1->copyToHostTensor(&tensor_scores_host1);
//    auto scores_dataPtr1 = tensor_scores_host1.host<float>();
//
//    char* score_layer_name2 ="layer125-conv";  //layer79-conv  20*20
//    MNN::Tensor* tensor_scores2 = netPtr_shoulderDetect->getSessionOutput(sessionPtr_shoulderDetect, score_layer_name2);
//    MNN::Tensor tensor_scores_host2(tensor_scores2, tensor_scores2->getDimensionType());
//    tensor_scores2->copyToHostTensor(&tensor_scores_host2);
//    auto scores_dataPtr2 = tensor_scores_host2.host<float>();
//
//    // 检测输入和输出大小是否正确
//    std::vector<int> a = tensor_scores1->shape();
//    std::vector<int> b = tensor_scores2->shape();
//    std::vector<int> c = tensorPtr->shape();
//
//    float anchorSize2[6] = {25, 29,  63, 61,  89,96};  //personDetect_v4_2020_0918
//    float anchorSize1[6] = {132, 144, 195,194, 299,300};
//    int gridSize1 = 10;
//    int gridSize2 = 20;
//
//    if( flag==2 )
//    {
//        gridSize1 = 5;
//        gridSize2 = 10;
//
//        anchorSize2[0] = 13; anchorSize2[1] = 15;
//        anchorSize2[2] = 32; anchorSize2[3] = 31;
//        anchorSize2[4] = 45; anchorSize2[5] = 48;
//
//        anchorSize1[0] = 66; anchorSize1[1] = 72;
//        anchorSize1[2] = 98; anchorSize1[3] = 97;
//        anchorSize1[4] = 150; anchorSize1[5] = 150;
//    }
//
//    int anchorNum = 3;
//    int featureC = 6;  //6 is (x y w h score clsScore)
//    int k = 0;
//    int gridNum2 = gridSize2*gridSize2;
//    int gridNum1 = gridSize1*gridSize1;
//    float scoreThreshold = 0.9;
//
//    // feature 20*20 远处目标
//    std::vector<shoulderRect> tmp_shoulders2;
//    for(int i=0; i<gridSize2; i++)  //h
//    {
//        for(int j=0; j<gridSize2; j++)  //w
//        {
//            for(int m=0; m<anchorNum*featureC; m++) // 6 is (x y w h score clsScore)
//            {
//                if(m%featureC == 4) // score
//                {
//                    k = m/featureC;
//                    float score = mnnSD_sigmoid(scores_dataPtr2[m*gridNum2+i*gridSize2+j]);
//                     if( score > scoreThreshold )  // 前景和背景的分类得分
//                    {
//                        // 仅有一类所以不需要对分类得分进行排序  centerx centery w h -> x y w h
//                        shoulderRect tmp_shoulder1;
//                        tmp_shoulder1.x = (int)((j + mnnSD_sigmoid(scores_dataPtr2[(m-4)*gridNum2+i*gridSize2+j])) / gridSize2 * in_h);
//                        tmp_shoulder1.y = (int)((i + mnnSD_sigmoid(scores_dataPtr2[(m-3)*gridNum2+i*gridSize2+j])) / gridSize2 * in_h);
//                        tmp_shoulder1.width  = (int)(exp(scores_dataPtr2[(m-2)*gridNum2+i*gridSize2+j]) * anchorSize2[2*k] / (float)in_w * in_w);
//                        tmp_shoulder1.height = (int)(exp(scores_dataPtr2[(m-1)*gridNum2+i*gridSize2+j]) * anchorSize2[2*k+1] / (float)in_h * in_h);
//                        tmp_shoulder1.x = tmp_shoulder1.x - 0.5*tmp_shoulder1.width;
//                        tmp_shoulder1.y = tmp_shoulder1.y - 0.5*tmp_shoulder1.height;
//                        tmp_shoulder1.score = score;
//
//                        tmp_shoulders2.push_back(tmp_shoulder1);
//                    }
//                }
//            }
//        }
//    }
//
//    std::sort(tmp_shoulders2.begin(), tmp_shoulders2.end(), [](const shoulderRect a, const shoulderRect b) { return a.score > b.score; });
//    // opt nms　1
//    std::vector<cv::Rect> shoulders2;
//    float nms_threshold2 = 0.3;
//    int N2 = tmp_shoulders2.size();
//    std::vector<int> labels2(N2, -1);
//    for(int i = 0; i < N2-1; ++i)
//    {
//        for (int j = i+1; j < N2; ++j)
//        {
//            shoulderRect pre_box = tmp_shoulders2[i];
//            shoulderRect cur_box = tmp_shoulders2[j];
//            float iou_ = iouShoulder(pre_box, cur_box);
//            if (iou_ > nms_threshold2) {
//                labels2[j] = 0;
//            }
//        }
//    }
//    for (int i = 0; i < N2; ++i)
//    {
//        if (labels2[i] == -1)
//        {
//           cv::Rect pre_box;
//           pre_box.x = tmp_shoulders2[i].x;
//           pre_box.y = tmp_shoulders2[i].y;
//           pre_box.width = tmp_shoulders2[i].width;
//           pre_box.height = tmp_shoulders2[i].height;
//           shoulders2.push_back(pre_box);   // 返回输出的hands
//        }
//        //persons2.push_back(tmp_persons2[i]);   // 返回输出的hands
//    }
//
//    // feature 10*10 近处目标
//    std::vector<shoulderRect> tmp_shoulders1;
//    for(int i=0; i<gridSize1; i++)  //h
//    {
//        for(int j=0; j<gridSize1; j++)  //w
//        {
//            for(int m=0; m<anchorNum*featureC; m++) // 6 is (x y w h score clsScore)
//            {
//                if(m%featureC == 4) // score
//                {
//                    k = m/featureC;
//                    float score = mnnSD_sigmoid(scores_dataPtr1[m*gridNum1+i*gridSize1+j]);
//                    // if( scores_dataPtr2[m*gridNum2+i*gridSize2+j] > scoreThreshold )  // 前景和背景的分类得分
//                     if( score > scoreThreshold )  // 前景和背景的分类得分
//                    {
//                        // 仅有一类所以不需要对分类得分进行排序  centerx centery w h -> x y w h
//                        shoulderRect tmp_shoulder1;
//                        tmp_shoulder1.x = (int)((j + mnnSD_sigmoid(scores_dataPtr1[(m-4)*gridNum1+i*gridSize1+j])) / gridSize1 * in_w);
//                        tmp_shoulder1.y = (int)((i + mnnSD_sigmoid(scores_dataPtr1[(m-3)*gridNum1+i*gridSize1+j])) / gridSize1 * in_h);
//                        tmp_shoulder1.width  = (int)(exp(scores_dataPtr1[(m-2)*gridNum1+i*gridSize1+j]) * anchorSize1[2*k] / (float)in_w * in_h);
//                        tmp_shoulder1.height = (int)(exp(scores_dataPtr1[(m-1)*gridNum1+i*gridSize1+j]) * anchorSize1[2*k+1] / (float)in_h * in_h);
//                        tmp_shoulder1.x = tmp_shoulder1.x - 0.5*tmp_shoulder1.width;
//                        tmp_shoulder1.y = tmp_shoulder1.y - 0.5*tmp_shoulder1.height;
//                        tmp_shoulder1.score = score;
//
//                        tmp_shoulders1.push_back(tmp_shoulder1);
//                        //LOGD("score is %f, %f",scores_dataPtr1[m*gridNum1+i*gridSize1+j], score);
//                        //LOGD("hand box is %f, %f, %f, %f",(float)tmp_hand.x, (float)tmp_hand.y, (float)tmp_hand.width, (float)tmp_hand.height);
//                    }
//                }
//            }
//        }
//    }
//    std::sort(tmp_shoulders1.begin(), tmp_shoulders1.end(), [](const shoulderRect a, const shoulderRect b) { return a.score > b.score; });
//
//    // opt nms　2
//    std::vector<cv::Rect> shoulders1;
//    float nms_threshold1 = 0.3;
//    int N1 = tmp_shoulders1.size();
//    std::vector<int> labels1(N1, -1);
//    for(int i = 0; i < N1-1; ++i)
//    {
//        for (int j = i+1; j < N1; ++j)
//        {
//            shoulderRect pre_box = tmp_shoulders1[i];
//            shoulderRect cur_box = tmp_shoulders1[j];
//            float iou_ = iouShoulder(pre_box, cur_box);
//            if (iou_ > nms_threshold1) {
//                labels1[j] = 0;
//            }
//        }
//    }
//    for (int i = 0; i < N1; ++i)
//    {
//        if (labels1[i] == -1)
//        {
//            cv::Rect pre_box;
//            pre_box.x = tmp_shoulders1[i].x;
//            pre_box.y = tmp_shoulders1[i].y;
//            pre_box.width = tmp_shoulders1[i].width;
//            pre_box.height = tmp_shoulders1[i].height;
//            shoulders2.push_back(pre_box);   // 返回输出的hands
//        }
//    }
//
//    // opt nms　3
//    //std::vector<cv::Rect> persons1;
//    float nms_threshold = 0.3;
//    int N = shoulders2.size();
//    std::vector<int> labels(N, -1);
//    for(int i = 0; i < N-1; ++i)
//    {
//        for (int j = i+1; j < N; ++j)
//        {
//            cv::Rect pre_box = shoulders2[i];
//            cv::Rect cur_box = shoulders2[j];
//            float iou_ = iouShoulder1(pre_box, cur_box);
//            if (iou_ > nms_threshold) {
//                labels[j] = 0;
//            }
//        }
//    }
//    for (int i = 0; i < N; ++i)
//    {
//        if (labels[i] == -1)
//            shoulders.push_back(shoulders2[i]);   // 返回输出的hands
//    }
//    return 0;
//}
