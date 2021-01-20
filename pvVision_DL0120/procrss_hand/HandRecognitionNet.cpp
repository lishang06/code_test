#include "handRecognition.h"

Inference_engineHR::Inference_engineHR()
{ }

Inference_engineHR::~Inference_engineHR()
{ 
    if ( netPtr_HandRecognition != NULL )
	{
		if ( sessionPtr_HandRecognition != NULL)
		{
			netPtr_HandRecognition->releaseSession(sessionPtr_HandRecognition);
			sessionPtr_HandRecognition = NULL;
		}

		delete netPtr_HandRecognition;
		netPtr_HandRecognition = NULL;
	}
}

int Inference_engineHR::load_param(std::vector<std::string>& file, int num_thread)
{
    if (!file.empty())
    {
        if (file[0].find(".mnn") != std::string::npos)
        {
	        netPtr_HandRecognition = MNN::Interpreter::createFromFile(file[0].c_str());
            if (nullptr == netPtr_HandRecognition) 
            {
                while(1)
                {

                }
                return -1;
            }
			
			//Ĭ������MNNForwardType ΪCPU
            MNN::ScheduleConfig sch_config;
            sch_config.type = (MNNForwardType)MNN_FORWARD_CPU;   // MNN_FORWARD_METAL
            //sch_config.type = (MNNForwardType)MNN_FORWARD_OPENCL;
            //sch_config.type = (MNNForwardType)MNN_FORWARD_OPENGL;
            //sch_config.type = (MNNForwardType)MNN_FORWARD_VULKAN;

            MNN::BackendConfig backendConfig;
            backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;
            sch_config.backendConfig = &backendConfig;

            num_thread = 4;
            if ( num_thread > 0 )sch_config.numThread = num_thread;
            sessionPtr_HandRecognition = netPtr_HandRecognition->createSession(sch_config);
            if (nullptr == sessionPtr_HandRecognition) 
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


int Inference_engineHR::HandRecognition(cv::Mat& img)
{
    MNN::Tensor* tensorPtr = netPtr_HandRecognition->getSessionInput(sessionPtr_HandRecognition, nullptr);
   
    // int in_h = 128;
    // int in_w = 128;
    int in_h = 112;
    int in_w = 112;
//    float mean_vals[3] = {0.0, 0.0, 0.0};
//    float norm_vals[3] = {1.0, 1.0, 1.0};
    float mean_vals[3] = {127.0, 127.0, 127.0};
    float norm_vals[3] = {0.0078125, 0.0078125, 0.0078125};
    netPtr_HandRecognition->resizeTensor(tensorPtr, {1, 3, in_h, in_w});
    netPtr_HandRecognition->resizeSession(sessionPtr_HandRecognition);
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(
            MNN::CV::ImageProcess::create(MNN::CV::RGB, MNN::CV::BGR, mean_vals, 3,
                                          norm_vals, 3));
    pretreat->convert(img.data, in_w, in_h, img.step[0], tensorPtr);

	netPtr_HandRecognition->runSession(sessionPtr_HandRecognition);

    char* score_layer_name ="softmax";
    MNN::Tensor* tensor_scores = netPtr_HandRecognition->getSessionOutput(sessionPtr_HandRecognition, score_layer_name);
    MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
    tensor_scores->copyToHostTensor(&tensor_scores_host);
    auto scores_dataPtr = tensor_scores_host.host<float>();

    // 显示维度信息
    //std::vector<int> a = tensor_scores->shape();
    //LOGD(" faceQ output is %d, %d, %d, %d",a[0], a[1], a[2], a[3]);

    /* 手势识别　0(0)大拇指, 2(1)负样本，4(2)剪刀手，　6(3)拍照，　8(4)五指张开 */
    // 获取最大分类得分
    float score[5];
    for(int i=0; i<5; i++)
    {
        score[i] = scores_dataPtr[i];
        //printf("i score is %d, %f\n",i, score[i]);
    }

    int maxId = -1;
    float maxScore = 0;
    for(int i=0; i<5; i++)
    {
        if( score[i]>maxScore )
        {
            maxScore = score[i];
            maxId = i;
        }
    }

    if( maxScore<0.8 )  maxId = 1;
    
    //printf("id is %d\n", maxId);


    return maxId;
}
