#include "handDetect.h"
#include "fstream"
#include "iostream"
#include "string"
using namespace std;

float iouHand(handRect box0, handRect box1)
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

float iouHand1(cv::Rect box0, cv::Rect box1)
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

Inference_engineHD::Inference_engineHD()
{ }

Inference_engineHD::~Inference_engineHD()
{ 
    if ( netPtr_handDetect != NULL )
	{
		if ( sessionPtr_handDetect != NULL)
		{
			netPtr_handDetect->releaseSession(sessionPtr_handDetect);
			sessionPtr_handDetect = NULL;
		}

		delete netPtr_handDetect;
		netPtr_handDetect = NULL;
	}
}

int Inference_engineHD::load_param(std::vector<std::string>& file, int num_thread)
{
    if (!file.empty())
    {
        if (file[0].find(".mnn") != std::string::npos)
        {
	        netPtr_handDetect = MNN::Interpreter::createFromFile(file[0].c_str());
            if (nullptr == netPtr_handDetect) 
            {
                while(1)
                {

                }
                return -1;
            }
			
			//Ĭ������MNNForwardType ΪCPU
            MNN::ScheduleConfig sch_config;
            sch_config.type = (MNNForwardType)MNN_FORWARD_CPU;  //MNN_FORWARD_METAL
            //sch_config.type = (MNNForwardType)MNN_FORWARD_OPENCL;
            //sch_config.type = (MNNForwardType)MNN_FORWARD_OPENGL;
            //sch_config.type = (MNNForwardType)MNN_FORWARD_VULKAN;

            MNN::BackendConfig backendConfig;
            backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;
            sch_config.backendConfig = &backendConfig;

            num_thread = 4;
            if ( num_thread > 0 )sch_config.numThread = num_thread;
            sessionPtr_handDetect = netPtr_handDetect->createSession(sch_config);
            if (nullptr == sessionPtr_handDetect) 
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


int Inference_engineHD::handDetect_infer_img(cv::Mat& img, std::vector<cv::Rect> &hands)
{
    MNN::Tensor* tensorPtr = netPtr_handDetect->getSessionInput(sessionPtr_handDetect, nullptr);
   
    int in_h = 176;
    int in_w = 176;
    float mean_vals[3] = {0.0, 0.0, 0.0};
    float norm_vals[3] = {1.0/255.0, 1.0/255.0, 1.0/255.0};
    netPtr_handDetect->resizeTensor(tensorPtr, {1, 3, in_h, in_w});
    netPtr_handDetect->resizeSession(sessionPtr_handDetect);
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(
            MNN::CV::ImageProcess::create(MNN::CV::RGB, MNN::CV::RGB, mean_vals, 3,
                                          norm_vals, 3));
    pretreat->convert(img.data, in_w, in_h, img.step[0], tensorPtr);

	netPtr_handDetect->runSession(sessionPtr_handDetect);

    char* score_layer_name ="layer53-conv";  //layer52-conv　　layer53-conv
    MNN::Tensor* tensor_scores = netPtr_handDetect->getSessionOutput(sessionPtr_handDetect, score_layer_name);
    MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
    tensor_scores->copyToHostTensor(&tensor_scores_host);
    auto scores_dataPtr = tensor_scores_host.host<float>();

    // 检测输入和输出大小是否正确
    std::vector<int> a = tensor_scores->shape();
    std::vector<int> c = tensorPtr->shape();

    //printf("start hand detect \n");
    
    /* 检测输出结果的预处理 */
    // anchor bias
//    float anchorSize[10] = {5,5,12,11,22,21,36,35,75,71};
    float anchorSize[10] = {7,7,14,14,26,25,42,38,90,84};
    int gridSize = 11;
    int k = 0;
    int gridNum = 11*11;
    float scoreThreshold = 0.45; //0.9
    std::vector<cv::Rect> tmp_hands;
    std::vector<handRect> tmp_hands_tmp;
    for(int i=0; i<gridSize; i++)  //h
    {
        for(int j=0; j<gridSize; j++)  //w
        {
            for(int m=0; m<5*6; m++) // 6 is (x y w h score clsScore)
            {
                if(m%6 == 4) // score
                {
                    k = m/6; 
                    float score = mnn_sigmoid(scores_dataPtr[m*gridNum+i*gridSize+j]);
                    //printf("score is %f\n",score);
                    // if( scores_dataPtr[m*gridNum+i*16+j] > scoreThreshold )  // 前景和背景的分类得分
                    if( score > scoreThreshold )  // 前景和背景的分类得分
                    {
                        //printf("have hand\n");
                        // 仅有一类所以不需要对分类得分进行排序  centerx centery w h -> x y w h
                        handRect tmp_hand;
                        tmp_hand.x = (int)((j + mnn_sigmoid(scores_dataPtr[(m-4)*gridNum+i*gridSize+j])) / gridSize * img.cols);
                        tmp_hand.y = (int)((i + mnn_sigmoid(scores_dataPtr[(m-3)*gridNum+i*gridSize+j])) / gridSize * img.rows);
                        tmp_hand.width  = (int)(exp(scores_dataPtr[(m-2)*gridNum+i*gridSize+j]) * anchorSize[2*k] / 176.0 * img.cols);
                        tmp_hand.height = (int)(exp(scores_dataPtr[(m-1)*gridNum+i*gridSize+j]) * anchorSize[2*k+1] / 176.0 * img.rows);
                        tmp_hand.x = tmp_hand.x - 0.5*tmp_hand.width;
                        tmp_hand.y = tmp_hand.y - 0.5*tmp_hand.height;
                        
                        //printf("tmp_hand is %d, %d, %d, %d\n", tmp_hand.x ,tmp_hand.y ,tmp_hand.width ,tmp_hand.height );

                        //tmp_hands.push_back(tmp_hand);
                        tmp_hands_tmp.push_back(tmp_hand);
                    }
                }
            }
        }
    }

    // sort
    std::sort(tmp_hands_tmp.begin(), tmp_hands_tmp.end(), [](const handRect a, const handRect b) { return a.score > b.score; });
    
    // opt nms　
    float nms_threshold = 0.3;
    int N = tmp_hands_tmp.size();
    std::vector<int> labels(N, -1);
    for(int i = 0; i < N-1; ++i)
    {
        for (int j = i+1; j < N; ++j)
        {
            handRect pre_box = tmp_hands_tmp[i];
            handRect cur_box = tmp_hands_tmp[j];
            float iou_ = iouHand(pre_box, cur_box);
            if (iou_ > nms_threshold) {
                labels[j] = 0;
            }
        }
    }
    for (int i = 0; i < N; ++i)
    {
        if (labels[i] == -1)
        {
            cv::Rect pre_box;
            pre_box.x = tmp_hands_tmp[i].x;
            pre_box.y = tmp_hands_tmp[i].y;
            pre_box.width = tmp_hands_tmp[i].width;
            pre_box.height = tmp_hands_tmp[i].height;
            hands.push_back(pre_box);   // 返回输出的hands
            //printf("tmp_hands is %d, %d, %d, %d\n", tmp_hands_tmp[i].x ,tmp_hands_tmp[i].y ,tmp_hands_tmp[i].width ,tmp_hands_tmp[i].height);
        }
    }
    return 0;
}
