#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "boost/scoped_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/loss_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {
  template <typename Dtype>
  inline Dtype logExp(Dtype x) {
    if(typeid(x)==typeid(float) && x>Dtype(88.0)){
      printf("float: %f, ",x);
      return x;
    }
    else if(typeid(x)==typeid(double) && x>Dtype(709.0)){
      printf("double: %f, ",x);
      return x;
    }
    else{
      return log(1.0+exp(x));
    }
  }
    template <typename TypeParam>
    class SemiPairLossLayerTest : public MultiDeviceTest<TypeParam> {
        typedef typename TypeParam::Dtype Dtype;
        protected:
        SemiPairLossLayerTest()
            : blob_bottom_data_(new Blob<Dtype>(300,10,1,1)),
              blob_bottom_y_(new Blob<Dtype>(300,1,1,1)),
              blob_top_loss_(new Blob<Dtype>()) {

              FillerParameter filler_param;
              //UniformFiller<Dtype> filler(filler_param);
              GaussianFiller<Dtype> filler(filler_param);
              filler.Fill(this->blob_bottom_data_);
              blob_bottom_vec_.push_back(blob_bottom_data_);
              int ncount = 0;
              printf("lebel:");
              for (int i = 0;i<blob_bottom_y_->count();i++) {
                  blob_bottom_y_->mutable_cpu_data()[i] = caffe_rng_rand() % 5-Dtype(0.0);
                  if(blob_bottom_y_->cpu_data()[i]==Dtype(-1))
                    ncount++;
                  printf("%.0f ",blob_bottom_y_->cpu_data()[i]);
              }
              printf(":num:%d\n",ncount);
              blob_bottom_vec_.push_back(blob_bottom_y_);
              blob_top_vec_.push_back(blob_top_loss_);
            }
        virtual ~SemiPairLossLayerTest() {
            delete blob_bottom_data_;
            delete blob_bottom_y_;
            delete blob_top_loss_;
        }

        Blob<Dtype>* const blob_bottom_data_;
        Blob<Dtype>* const blob_bottom_y_;
        Blob<Dtype>* const blob_top_loss_;
        vector<Blob<Dtype>*> blob_bottom_vec_;
        vector<Blob<Dtype>*> blob_top_vec_;
    };
    template <typename Dtype>
      struct SimiNode{
      Dtype simi;
      int index;
      SimiNode(Dtype value,int idx){
        simi = value;
        index = idx;
      }
      friend bool operator<(const SimiNode& sn1,const SimiNode& sn2){
        return sn1.simi>=sn2.simi;
      }
    };
    TYPED_TEST_CASE(SemiPairLossLayerTest, TestDtypesAndDevices);
    TYPED_TEST(SemiPairLossLayerTest, TestForward) {
      typedef typename TypeParam::Dtype Dtype;
      LayerParameter layer_param;
      SemiPairLossLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      
      // manually compute to compare
      int num = this->blob_bottom_data_->num();
      int count = this->blob_bottom_data_->count();
      int dim = count/num;
      Blob<Dtype> blob_simi_s(num,num,1,1);
      Blob<Dtype> blob_theta(num,num,1,1);
      
      const Dtype* bottom_data = this->blob_bottom_data_->cpu_data();
      const Dtype* bottom_label = this->blob_bottom_y_->cpu_data();
      
      Dtype* simi_s_data = blob_simi_s.mutable_cpu_data();
      Dtype* theta_data = blob_theta.mutable_cpu_data();
      //calculate matrix S
      for(int i = 0; i<num; i++){
        vector<SimiNode<Dtype> > simi_vec;
        Blob<Dtype> diff(dim,1,1,1);
        for(int j=0;j<num;j++){
          if(bottom_label[i]==Dtype(-1.0)||bottom_label[j]==Dtype(-1.0)){
            caffe_sub(dim,bottom_data+i*dim,bottom_data+j*dim,diff.mutable_cpu_data());
            Dtype dist = caffe_cpu_dot(dim,diff.cpu_data(),diff.cpu_data());
            simi_vec.push_back(SimiNode<Dtype>(-dist,j));
          }
          else{
            if (bottom_label[i] == bottom_label[j]){
              simi_s_data[i*num+j] = Dtype(1.0);
            }
            else{
              simi_s_data[i*num+j] = Dtype(0.0);
            }
          }

        }
        std::sort(simi_vec.begin(),simi_vec.end());
        int value_k = std::min(layer_param.semipair_loss_param().knn_k(),static_cast<int>(simi_vec.size()));
        //CHECK_EQ(static_cast<int>(simi_vec.size()),0);
        CHECK_EQ(layer_param.semipair_loss_param().knn_k(),10);

        for(int k=0;k<value_k;k++){
          if(k>0){
            CHECK_LE(simi_vec[k].simi,simi_vec[k-1].simi);
          }
          if(i==0)
            printf("%f ",simi_vec[k].simi);
          simi_s_data[i*num+simi_vec[k].index] = Dtype(1.0);
        }
        for(int k = value_k;k<simi_vec.size();k++){
          CHECK_LE(simi_vec[k].simi,simi_vec[k-1].simi);
          simi_s_data[i*num+simi_vec[k].index] = Dtype(0.0);
        }
        if(i==0) printf("\n");
      }
      // force symmetric
      for(int i=0;i<num;i++){
        for(int j=0;j<=i;j++){
          if(simi_s_data[i*num+j]==Dtype(1.0)||simi_s_data[j*num+i]==Dtype(1.0)){
            simi_s_data[i*num+j] = Dtype(1.0);
            simi_s_data[j*num+i] = Dtype(1.0);
          }
        }
      }
      //calculate loss
      Dtype loss(0.0);
      for (int i = 0; i < num; i++) {
        for (int j = 0; j < num; j++) {
          theta_data[i*num + j] = Dtype(0.5)*caffe_cpu_dot(dim, bottom_data+i*dim, bottom_data+j*dim);
          if (blob_simi_s.cpu_data()[i*num+j] == Dtype(1.0)){
            //loss -= (blob_theta.cpu_data()[i*num+j] - log(Dtype(1.0)+exp(blob_theta.cpu_data()[i*num+j])));
            loss -= (blob_theta.cpu_data()[i*num+j] - logExp(blob_theta.cpu_data()[i*num+j])); 
          }
          else{
            //loss += log(Dtype(1.0)+exp(blob_theta.cpu_data()[i*num+j]));
            //loss += logExp(blob_theta.cpu_data()[i*num+j]);
          }
        }
      }
      loss = loss / static_cast<Dtype>(num);
      EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-6);
    }
    TYPED_TEST(SemiPairLossLayerTest, TestGradient) {
        typedef typename TypeParam::Dtype Dtype;
        LayerParameter layer_param;
        SemiPairLossLayer<Dtype> layer(layer_param);
        layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        GradientChecker<Dtype> checker(1e-2, 1e-2, 1071);
        printf("SemiPairLoss: Begin check Gradient\n");
        checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
    }
    
} // namespace caffe
