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
    class SemiContrastLossLayerTest : public MultiDeviceTest<TypeParam> {
        typedef typename TypeParam::Dtype Dtype;
        protected:
        SemiContrastLossLayerTest()
            : blob_bottom_data_(new Blob<Dtype>(300,10,1,1)),
              blob_bottom_y_(new Blob<Dtype>(300,1,1,1)),
              blob_bottom_feat_(new Blob<Dtype>(300,10,1,1)),
              blob_top_loss_(new Blob<Dtype>()) {

              FillerParameter filler_param;
              UniformFiller<Dtype> filler(filler_param);
              //GaussianFiller<Dtype> filler(filler_param);
              filler.Fill(this->blob_bottom_data_);
              blob_bottom_vec_.push_back(blob_bottom_data_);
              const Dtype* data = blob_bottom_data_->cpu_data();
              for(int i=0;i<blob_bottom_data_->count();i++){
                printf("%.3f ",data[i]);
                if((i+1)%10==0) printf("\n");
              }
              printf("\n");
              int ncount = 0;
              printf("label:");
              for (int i = 0;i<blob_bottom_y_->count();i++) {
                  blob_bottom_y_->mutable_cpu_data()[i] = caffe_rng_rand() % 10-Dtype(1.0);
                  if(blob_bottom_y_->cpu_data()[i]==Dtype(-1))
                    ncount++;
                  printf("%.0f ",blob_bottom_y_->cpu_data()[i]);
              }
              printf(":num:%d\n",ncount);
              blob_bottom_vec_.push_back(blob_bottom_y_);
              filler.Fill(this->blob_bottom_feat_);
              blob_bottom_vec_.push_back(blob_bottom_feat_);
              blob_top_vec_.push_back(blob_top_loss_);
            }
        virtual ~SemiContrastLossLayerTest() {
            delete blob_bottom_data_;
            delete blob_bottom_y_;
            delete blob_bottom_feat_;
            delete blob_top_loss_;
        }

        Blob<Dtype>* const blob_bottom_data_;
        Blob<Dtype>* const blob_bottom_y_;
        Blob<Dtype>* const blob_bottom_feat_;
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
    TYPED_TEST_CASE(SemiContrastLossLayerTest, TestDtypesAndDevices);
    
    TYPED_TEST(SemiContrastLossLayerTest, TestGradient) {
        typedef typename TypeParam::Dtype Dtype;
        LayerParameter layer_param;
        SemiContrastLossLayer<Dtype> layer(layer_param);
        printf("TestGradient----------------------%f,%f,%d\n",layer_param.semicontrast_loss_param().margin(),
          layer_param.semicontrast_loss_param().lambda(),
          layer_param.semicontrast_loss_param().feature_index());
        layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        
        GradientChecker<Dtype> checker(1e-2, 1e-2, 1071);
        printf("SemiContrastLoss: Begin check Gradient\n");
        checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
    }
    
} // namespace caffe
