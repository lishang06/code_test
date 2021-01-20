#include "UltraFacenet.h"
#include "fstream"
#include "iostream"
#include "string"
using namespace std;

float iouFace(faceRect box0, faceRect box1)
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

Inference_engine::Inference_engine()
{ }

Inference_engine::~Inference_engine()
{ 
    if ( netPtr != NULL )
	{
		if ( sessionPtr != NULL)
		{
			netPtr->releaseSession(sessionPtr);
			sessionPtr = NULL;
		}

		delete netPtr;
		netPtr = NULL;
	}
}

int Inference_engine::load_param(std::vector<std::string>& file, int num_thread)
{
    if (!file.empty())
    {
        if (file[0].find(".mnn") != std::string::npos)
        {
	        netPtr = MNN::Interpreter::createFromFile(file[0].c_str());
            if (nullptr == netPtr) 
            {
                while(1)
                {

                }
                return -1;
            }
			
            MNN::ScheduleConfig sch_config;
            sch_config.type = (MNNForwardType)MNN_FORWARD_CPU;

            MNN::BackendConfig backendConfig;
            backendConfig.precision = (MNN::BackendConfig::Precision_Normal);
            sch_config.backendConfig = &backendConfig;

            num_thread = 4;
            if ( num_thread > 0 )sch_config.numThread = num_thread;
            sessionPtr = netPtr->createSession(sch_config);
            if (nullptr == sessionPtr) 
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

int Inference_engine::set_params(int srcType, int dstType, 
                                 std::vector<float>& mean, std::vector<float>& scale)
{
    config.destFormat   = (MNN::CV::ImageFormat)dstType;
    config.sourceFormat = (MNN::CV::ImageFormat)srcType;

    ::memcpy(config.mean,   &mean[0],   3 * sizeof(float));
    ::memcpy(config.normal, &scale[0],  3 * sizeof(float));

    config.filterType = (MNN::CV::Filter)(1);
    config.wrap = (MNN::CV::Wrap)(2);

    return 0;
}

// infer
int Inference_engine::infer_img(cv::Mat& img, Inference_engine_tensor& out)
{
    MNN::Tensor* tensorPtr = netPtr->getSessionInput(sessionPtr, nullptr);

    // auto resize for full conv network.
    bool auto_resize = false;
    if ( !auto_resize )
    {
        std::vector<int>dims = { 1, img.channels(), img.rows, img.cols };
        netPtr->resizeTensor(tensorPtr, dims);
        netPtr->resizeSession(sessionPtr);
    }

    // config.type = MNN_FORWARD_CPU;
	// config.numThread = num_thread;
    std::unique_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(config));
    process->convert((const unsigned char*)img.data, img.cols, img.rows, img.step[0], tensorPtr);
    
	// �������
	netPtr->runSession(sessionPtr);

	// ��ȡ�������
    for (int i = 0; i < out.layer_name.size(); i++)
    {
        const char* layer_name = NULL;
        if( strcmp(out.layer_name[i].c_str(), "") != 0)
        {
            layer_name = out.layer_name[i].c_str();
        }
        MNN::Tensor* tensorOutPtr = netPtr->getSessionOutput(sessionPtr, layer_name);

        std::vector<int> shape = tensorOutPtr->shape();
        cv::Mat feat(shape.size(), &shape[0], CV_32F);

        auto tensor = reinterpret_cast<MNN::Tensor*>(tensorOutPtr);
        float *destPtr = (float*)feat.data;
        if (nullptr == destPtr)
        {
            std::unique_ptr<MNN::Tensor> hostTensor(new MNN::Tensor(tensor, tensor->getDimensionType(), false));
            return hostTensor->elementSize();
        }

        std::unique_ptr<MNN::Tensor> hostTensor(new MNN::Tensor(tensor, tensor->getDimensionType(), true));
        tensor->copyToHostTensor(hostTensor.get());
        tensor = hostTensor.get();

        auto size = tensor->elementSize();
        ::memcpy(destPtr, tensor->host<float>(), size * sizeof(float));

        out.out_feat.push_back(feat.clone());
    }

    return 0;
}

int Inference_engine::infer_imgs(std::vector<cv::Mat>& imgs, std::vector<Inference_engine_tensor>& out)
{
    for (int i = 0; i < imgs.size(); i++)
    {
        infer_img(imgs[i], out[i]);
    }

    return 0;
}

float iou(cv::Rect box0, cv::Rect box1) 
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


struct timespec time11={0,0};
struct timespec time22={0,0};

int Inference_engine::Ultra_infer_img(cv::Mat& img,float conf_threshold,float nms_threshold,int OUTPUT_NUM,float center_variance,float size_variance,float anchors[4][4420],
std::vector<cv::Rect> &faces, std::vector<landmarkFace> &landmarkBoxResult)
{
    MNN::Tensor* tensorPtr = netPtr->getSessionInput(sessionPtr, nullptr);

    int in_h = 240;
    int in_w = 320;
    float mean_vals[3] = {0.0, 0.0, 0.0};
    float norm_vals[3] = {1/255.0, 1/255.0, 1/255.0};
    netPtr->resizeTensor(tensorPtr, {1, 3, in_h, in_w});
    
    netPtr->resizeSession(sessionPtr);

    
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(
            MNN::CV::ImageProcess::create(MNN::CV::RGB, MNN::CV::BGR, mean_vals, 3,
                                          norm_vals, 3));
    pretreat->convert(img.data, in_w, in_h, img.step[0], tensorPtr);
	
    clock_gettime(CLOCK_REALTIME, &time11);
    netPtr->runSession(sessionPtr);
    clock_gettime(CLOCK_REALTIME, &time22);
    printf("[MNN] time 1 is %d\n", (time22.tv_sec-time11.tv_sec)*1000 + (time22.tv_nsec-time11.tv_nsec)/1000000);


    char* score_layer_name ="scores";
    MNN::Tensor* tensor_scores = netPtr->getSessionOutput(sessionPtr, score_layer_name);
    MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
    tensor_scores->copyToHostTensor(&tensor_scores_host);
    auto scores_dataPtr = tensor_scores_host.host<float>();

    char* box_layer_name ="boxes";
    MNN::Tensor* tensor_boxes = netPtr->getSessionOutput(sessionPtr, box_layer_name);
    MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());
    tensor_boxes->copyToHostTensor(&tensor_boxes_host);
    auto boxes_dataPtr = tensor_boxes_host.host<float>();

    std::vector<cv::Rect> tmp_faces;
    std::vector<faceRect> tmp_faces_tmp;
    for(int i =0;i<OUTPUT_NUM;++i)
    {
        //float nonface_prob = scores_dataPtr[i*2 + 0];
        float face_prob    = scores_dataPtr[i*2 + 1];

        if (face_prob > conf_threshold)
        {
            //printf("[MNN] face_prob score is %f\n",face_prob);
            float xcenter =     boxes_dataPtr[i*4 + 0] * center_variance * anchors[2][i] + anchors[0][i];
            float ycenter =     boxes_dataPtr[i*4 + 1] * center_variance * anchors[3][i] + anchors[1][i];
            float w       = exp(boxes_dataPtr[i*4 + 2] * size_variance) * anchors[2][i];
            float h       = exp(boxes_dataPtr[i*4 + 3] * size_variance) * anchors[3][i];

            float ymin    = ( ycenter - h * 0.5 ) * img.rows;
            float xmin    = ( xcenter - w * 0.5 ) * img.cols;
            float ymax    = ( ycenter + h * 0.5 ) * img.rows;
            float xmax    = ( xcenter + w * 0.5 ) * img.cols;
            
            faceRect tmp_face;
            tmp_face.x = (int)xmin;
            tmp_face.y = (int)ymin;
            tmp_face.width  = (int)(xmax - xmin);
            tmp_face.height = (int)(ymax - ymin);
            tmp_faces_tmp.push_back(tmp_face);
        }
    }
    
    // sort
    std::sort(tmp_faces_tmp.begin(), tmp_faces_tmp.end(), [](const faceRect a, const faceRect b) { return a.score > b.score; });
    
    // perform NMS
    int N = tmp_faces_tmp.size();
    std::vector<int> labels(N, -1); 
    for(int i = 0; i < N-1; ++i)
    {
        for (int j = i+1; j < N; ++j)
        {
            faceRect pre_box = tmp_faces_tmp[i];
            faceRect cur_box = tmp_faces_tmp[j];
            float iou_ = iouFace(pre_box, cur_box);
            if (iou_ > nms_threshold) {
                labels[j] = 0;
            }
        }
    }
    for (int i = 0; i < N; ++i)
    {
        if (labels[i] == -1)
        {
            cv::Rect tmp_faces11;
            tmp_faces11.x = tmp_faces_tmp[i].x;
            tmp_faces11.y = tmp_faces_tmp[i].y;
            tmp_faces11.width  = tmp_faces_tmp[i].width;
            tmp_faces11.height = tmp_faces_tmp[i].height;
            faces.push_back(tmp_faces11);
        }
    }

    return 0;
}

int Inference_engine::Ultra_infer_img_160(cv::Mat& img,float conf_threshold,float nms_threshold,int OUTPUT_NUM,float center_variance,float size_variance,float anchors[4][1118],
std::vector<cv::Rect> &faces, std::vector<landmarkFace> &landmarkBoxResult)
{
    MNN::Tensor* tensorPtr = netPtr->getSessionInput(sessionPtr, nullptr);

    int in_h = 120;
    int in_w = 160;
    float mean_vals[3] = {0.0, 0.0, 0.0};
    float norm_vals[3] = {1/255.0, 1/255.0, 1/255.0};
    netPtr->resizeTensor(tensorPtr, {1, 3, in_h, in_w});
    netPtr->resizeSession(sessionPtr);
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(
            MNN::CV::ImageProcess::create(MNN::CV::RGB, MNN::CV::BGR, mean_vals, 3,
                                          norm_vals, 3));
    pretreat->convert(img.data, in_w, in_h, img.step[0], tensorPtr);
    clock_gettime(CLOCK_REALTIME, &time11);
    netPtr->runSession(sessionPtr);
    clock_gettime(CLOCK_REALTIME, &time22);
    printf("[MNN] time 2 is %d\n", (time22.tv_sec-time11.tv_sec)*1000 + (time22.tv_nsec-time11.tv_nsec)/1000000);

    char* score_layer_name ="scores";
    MNN::Tensor* tensor_scores = netPtr->getSessionOutput(sessionPtr, score_layer_name);
    MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
    tensor_scores->copyToHostTensor(&tensor_scores_host);
    auto scores_dataPtr = tensor_scores_host.host<float>();

    char* box_layer_name ="boxes";
    MNN::Tensor* tensor_boxes = netPtr->getSessionOutput(sessionPtr, box_layer_name);
    MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());
    tensor_boxes->copyToHostTensor(&tensor_boxes_host);
    auto boxes_dataPtr = tensor_boxes_host.host<float>();

    std::vector<cv::Rect> tmp_faces;
    std::vector<faceRect> tmp_faces_tmp;
    for(int i =0;i<OUTPUT_NUM;++i)
    {
        //printf("[MNN] i is %d\n", i);
        //float nonface_prob = scores_dataPtr[i*2 + 0];
        float face_prob    = scores_dataPtr[i*2 + 1];

        if (face_prob > conf_threshold)
        {
            //printf("[MNN] face_prob score is %f\n",face_prob);
            float xcenter =     boxes_dataPtr[i*4 + 0] * center_variance * anchors[2][i] + anchors[0][i];
            float ycenter =     boxes_dataPtr[i*4 + 1] * center_variance * anchors[3][i] + anchors[1][i];
            float w       = exp(boxes_dataPtr[i*4 + 2] * size_variance) * anchors[2][i];
            float h       = exp(boxes_dataPtr[i*4 + 3] * size_variance) * anchors[3][i];

            float ymin    = ( ycenter - h * 0.5 ) * img.rows;
            float xmin    = ( xcenter - w * 0.5 ) * img.cols;
            float ymax    = ( ycenter + h * 0.5 ) * img.rows;
            float xmax    = ( xcenter + w * 0.5 ) * img.cols;
            
            faceRect tmp_face;
            tmp_face.x = (int)xmin;
            tmp_face.y = (int)ymin;
            tmp_face.width  = (int)(xmax - xmin);
            tmp_face.height = (int)(ymax - ymin);
            tmp_faces_tmp.push_back(tmp_face);
        }
    }
    
    // sort
    std::sort(tmp_faces_tmp.begin(), tmp_faces_tmp.end(), [](const faceRect a, const faceRect b) { return a.score > b.score; });
    
    // perform NMS
    int N = tmp_faces_tmp.size();
    std::vector<int> labels(N, -1);
    for(int i = 0; i < N-1; ++i)
    {
        for (int j = i+1; j < N; ++j)
        {
            faceRect pre_box = tmp_faces_tmp[i];
            faceRect cur_box = tmp_faces_tmp[j];
            float iou_ = iouFace(pre_box, cur_box);
            if (iou_ > nms_threshold) {
                labels[j] = 0;
            }
        }
    }
    for (int i = 0; i < N; ++i)
    {
        if (labels[i] == -1)
        {
            cv::Rect tmp_faces11;
            tmp_faces11.x = tmp_faces_tmp[i].x;
            tmp_faces11.y = tmp_faces_tmp[i].y;
            tmp_faces11.width  = tmp_faces_tmp[i].width;
            tmp_faces11.height = tmp_faces_tmp[i].height;
            faces.push_back(tmp_faces11);
        }
    }

    return 0;
}


int Inference_engine::Ultra_infer_img_80(cv::Mat& img,float conf_threshold,float nms_threshold,int OUTPUT_NUM,float center_variance,float size_variance,float anchors[4][298],
std::vector<cv::Rect> &faces, std::vector<landmarkFace> &landmarkBoxResult)
{
    MNN::Tensor* tensorPtr = netPtr->getSessionInput(sessionPtr, nullptr);

    int in_h = 60;
    int in_w = 80;
    float mean_vals[3] = {0.0, 0.0, 0.0};
    float norm_vals[3] = {1/255.0, 1/255.0, 1/255.0};
    netPtr->resizeTensor(tensorPtr, {1, 3, in_h, in_w});
    netPtr->resizeSession(sessionPtr);
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(
            MNN::CV::ImageProcess::create(MNN::CV::RGB, MNN::CV::BGR, mean_vals, 3,
                                          norm_vals, 3));
    pretreat->convert(img.data, in_w, in_h, img.step[0], tensorPtr);
    clock_gettime(CLOCK_REALTIME, &time11);
    netPtr->runSession(sessionPtr);
    clock_gettime(CLOCK_REALTIME, &time22);
    printf("[MNN] time 4 is %d\n", (time22.tv_sec-time11.tv_sec)*1000 + (time22.tv_nsec-time11.tv_nsec)/1000000);

    char* score_layer_name ="scores";
    MNN::Tensor* tensor_scores = netPtr->getSessionOutput(sessionPtr, score_layer_name);
    MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
    tensor_scores->copyToHostTensor(&tensor_scores_host);
    auto scores_dataPtr = tensor_scores_host.host<float>();

    char* box_layer_name ="boxes";
    MNN::Tensor* tensor_boxes = netPtr->getSessionOutput(sessionPtr, box_layer_name);
    MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());
    tensor_boxes->copyToHostTensor(&tensor_boxes_host);
    auto boxes_dataPtr = tensor_boxes_host.host<float>();

    std::vector<cv::Rect> tmp_faces;
    std::vector<faceRect> tmp_faces_tmp;
    for(int i =0;i<OUTPUT_NUM;++i)
    {
        //float nonface_prob = scores_dataPtr[i*2 + 0];
        float face_prob    = scores_dataPtr[i*2 + 1];
        
        //printf("[MNN] i is %d\n", i);

        if (face_prob > conf_threshold)
        {
            //printf("[MNN] face_prob score is %f\n",face_prob);
            float xcenter =     boxes_dataPtr[i*4 + 0] * center_variance * anchors[2][i] + anchors[0][i];
            float ycenter =     boxes_dataPtr[i*4 + 1] * center_variance * anchors[3][i] + anchors[1][i];
            float w       = exp(boxes_dataPtr[i*4 + 2] * size_variance) * anchors[2][i];
            float h       = exp(boxes_dataPtr[i*4 + 3] * size_variance) * anchors[3][i];

            float ymin    = ( ycenter - h * 0.5 ) * img.rows;
            float xmin    = ( xcenter - w * 0.5 ) * img.cols;
            float ymax    = ( ycenter + h * 0.5 ) * img.rows;
            float xmax    = ( xcenter + w * 0.5 ) * img.cols;
            
            faceRect tmp_face;
            tmp_face.x = (int)xmin;
            tmp_face.y = (int)ymin;
            tmp_face.width  = (int)(xmax - xmin);
            tmp_face.height = (int)(ymax - ymin);
            tmp_faces_tmp.push_back(tmp_face);
        }
    }
    
    // sort
    std::sort(tmp_faces_tmp.begin(), tmp_faces_tmp.end(), [](const faceRect a, const faceRect b) { return a.score > b.score; });
    
    // perform NMS
    int N = tmp_faces_tmp.size();
    std::vector<int> labels(N, -1);
    for(int i = 0; i < N-1; ++i)
    {
        for (int j = i+1; j < N; ++j)
        {
            faceRect pre_box = tmp_faces_tmp[i];
            faceRect cur_box = tmp_faces_tmp[j];
            float iou_ = iouFace(pre_box, cur_box);
            if (iou_ > nms_threshold) {
                labels[j] = 0;
            }
        }
    }
    for (int i = 0; i < N; ++i)
    {
        if (labels[i] == -1)
        {
            cv::Rect tmp_faces11;
            tmp_faces11.x = tmp_faces_tmp[i].x;
            tmp_faces11.y = tmp_faces_tmp[i].y;
            tmp_faces11.width  = tmp_faces_tmp[i].width;
            tmp_faces11.height = tmp_faces_tmp[i].height;
            faces.push_back(tmp_faces11);
        }
    }

    return 0;
}
