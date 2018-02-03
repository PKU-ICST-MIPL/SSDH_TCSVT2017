//Created by Ye Liu (E-mail: jourkliu@163.com) from Sun Yat-sen University @ 2014-12-26

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/loss_layers.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void SemiMLabelSoftmaxLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  label_cnt_.Reshape(bottom[0]->num(),1,1,1);
  CHECK_EQ(bottom[0]->channels(),bottom[1]->channels());
}

template <typename Dtype>
void SemiMLabelSoftmaxLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SemiMLabelSoftmaxLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  //softmax_bottom_vec_[0] = bottom[0];
  //softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  //const Dtype* prob_data = prob_.cpu_data();
  const Dtype* prob_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();

  caffe_set(label_cnt_.count(),Dtype(0.0),label_cnt_.mutable_cpu_data());
  Dtype* label_cnt = label_cnt_.mutable_cpu_data();

  int num = prob_.num();
  int dim = prob_.count() / num;
  Dtype loss = 0;
  int count = 0;
  for (int i = 0; i < num; ++i) {
    label_cnt[i] = caffe_cpu_dot(dim,label+i*dim,label+i*dim);
    if(label_cnt[i]==Dtype(0.0)) continue;
    for (int j = 0; j < dim; j++) {
  	  if(label[i*dim + j] > 0){
  		  loss += -log(max(prob_data[i * dim + j], Dtype(FLT_MIN)));
  	  }
    }
    count++;
  }
  //(*top)[0]->mutable_cpu_data()[0] = loss / num / dim;
  top[0]->mutable_cpu_data()[0] = loss / count;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SemiMLabelSoftmaxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  const Dtype* label_cnt = label_cnt_.cpu_data();
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    //const Dtype* prob_data = prob_.cpu_data();
    const Dtype* prob_data = bottom[0]->cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    //LOG(INFO) << "All Sample Num: " << num << " Sample Dim:" << dim;
	//LOG(INFO) << "FLT_MIN = " << FLT_MIN << " kLOG_THRESHOLD" << kLOG_THRESHOLD;
    Dtype loss = 0;
    int count=0;
    for (int i = 0; i < num; ++i) {
      //LOG(INFO) << "num: " << i;
      if(label_cnt[i]==Dtype(0.0)){
        caffe_set(dim,Dtype(0.0),bottom_diff+i*dim);
      }
      else{
        for (int j = 0; j < dim; ++j) {
          //LOG(INFO) << "label: " << label[i*dim + j] << " bottom_diff:" << bottom_diff[i * dim + j];
    	    if(label[i*dim + j] > 0){
            bottom_diff[i * dim + j] -= 1.0;
    		    loss += -log(max(prob_data[i * dim + j], Dtype(FLT_MIN)));
    		  }
        }
        count++;
      }
    }
    //LOG(INFO) << "loss = " << loss / num;
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    //LOG(INFO) << "loss_weight = " << loss_weight;
	//loss_weight equal to 1
    //caffe_scal(prob_.count(), loss_weight / num / dim, bottom_diff);
	  caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(SemiMLabelSoftmaxLossLayer);
#endif

INSTANTIATE_CLASS(SemiMLabelSoftmaxLossLayer);
REGISTER_LAYER_CLASS(SemiMLabelSoftmaxLoss);

}  // namespace caffe
