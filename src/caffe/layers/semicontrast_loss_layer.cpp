// Initial version
#include <algorithm>
#include <vector>
#include <typeinfo>
#include <float.h>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

   template <typename Dtype>
     void SemiContrastLossLayer<Dtype>::LayerSetUp(
       const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        LossLayer<Dtype>::LayerSetUp(bottom, top);
        int dim = bottom[0]->count()/bottom[0]->num();
        int count = bottom[0]->count();
        int num = bottom[0]->num();
        int fea_index = this->layer_param_.semicontrast_loss_param().feature_index();
        CHECK(fea_index==0||fea_index==2);
        LOG(INFO)<<"margin: "<<this->layer_param_.semicontrast_loss_param().margin()
          <<" margin_rank: "<<this->layer_param_.semicontrast_loss_param().margin_rank()
          <<" --- lambda: "<<this->layer_param_.semicontrast_loss_param().lambda()
          <<" --- feature_index: "<<this->layer_param_.semicontrast_loss_param().feature_index()<<"\n";
        LOG(INFO)<<" --- w_pair: "<<this->layer_param_.semicontrast_loss_param().weight()
          <<" --- w_rank: "<<this->layer_param_.semicontrast_loss_param().weight_rank()<<"\n";
        LOG(INFO)<<"count: "<<count<<" --- num: "<<num<<" --- dim: "<<dim<<"--- feature dim: "<<bottom[fea_index]->channels()<<"\n";
        CHECK_EQ(bottom[0]->channels(), dim);
        CHECK_EQ(bottom[0]->height(), 1);
        CHECK_EQ(bottom[0]->width(), 1);
        CHECK_EQ(bottom[1]->channels(), 1);
        CHECK_EQ(bottom[1]->height(), 1);
        CHECK_EQ(bottom[1]->width(), 1);
        CHECK_EQ(bottom[2]->num(),bottom[0]->num());
        CHECK_EQ(bottom[2]->height(),1);
        CHECK_EQ(bottom[2]->width(),1);
        simi_s.Reshape(num,num,1,1);
        dist_sq_.Reshape(num,num,1,1);
        fea_diff_.Reshape(bottom[fea_index]->channels(),1,1,1);
        data_diff_.Reshape(bottom[0]->channels(),1,1,1);
     }

   template <typename Dtype>
     void SemiContrastLossLayer<Dtype>::Forward_cpu(
       const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top) {
        printf("Forward_cpu... ");
        //int count = bottom[0]->count();
        int num = bottom[0]->num();
        int dim = bottom[0]->channels();
        int fea_index = this->layer_param_.semicontrast_loss_param().feature_index();
        //fea_index = 0;
        int fea_dim = bottom[fea_index]->channels();
        printf("fea_dim:%d -- ",fea_dim);
        
        //reset
        caffe_set(dist_sq_.count(),Dtype(0.0),dist_sq_.mutable_cpu_data());
        caffe_set(simi_s.count(),Dtype(-1.0),simi_s.mutable_cpu_data());
        vec_anchor.clear();
        vec_pos.clear();
        vec_neg.clear();

        const Dtype* bottom_data = bottom[0]->cpu_data(); // bottom_data is n*dim matrix
        const Dtype* bottom_label = bottom[1]->cpu_data();
        const Dtype* bottom_fea = bottom[fea_index]->cpu_data();
        
	Dtype* simi_s_data = simi_s.mutable_cpu_data();
        Dtype* dist_sq_data = dist_sq_.mutable_cpu_data();

        // Calculate S matrix online
        caffe_set(simi_s.count(),Dtype(-1.0),simi_s_data);

        for(int i = 0; i<num; i++){
           vector < pair<Dtype,int> > simi_vec;
           simi_vec.clear();
           vector<int> neg_idx_vec;
           vector<int> pos_idx_vec;
           for(int j=0;j<num;j++){

              //compute L2 similarity based on feature
              if (bottom_label[i] == Dtype(-1.0) || bottom_label[j] == Dtype(-1.0)) {
                 caffe_sub(fea_dim,bottom_fea+i*fea_dim,bottom_fea+j*fea_dim,fea_diff_.mutable_cpu_data());
                 Dtype fea_dist = caffe_cpu_dot(fea_dim,fea_diff_.cpu_data(),fea_diff_.cpu_data());
                 pair<Dtype, int> pitem(fea_dist,j);
                 simi_vec.push_back(pitem);
                 //simi_vec[j].first = fea_dist;
                 //simi_vec[j].second = j;
              }
              //compute data distance
              caffe_sub(dim,bottom_data+i*dim,bottom_data+j*dim,data_diff_.mutable_cpu_data());
              dist_sq_data[i*num+j] = caffe_cpu_dot(dim,data_diff_.cpu_data(),data_diff_.cpu_data());
              if(bottom_label[i]==Dtype(-1.0)||bottom_label[j]==Dtype(-1.0)){
              }
              else{
                 if (bottom_label[i] == bottom_label[j]){
                    simi_s_data[i*num+j] = Dtype(1.0);
                    pos_idx_vec.push_back(j);
                 }
                 else{
                    simi_s_data[i*num+j] = Dtype(0.0);
                    neg_idx_vec.push_back(j);
                 }
              }
           }
           std::sort(simi_vec.begin(),simi_vec.end());
           
           int vec_ind = 0;
           int index = 0;

           int length = static_cast<int>(simi_vec.size())/3;
           //select semi similar
           while(true){
              vec_ind = caffe_rng_rand()%std::min(10,length);
              index = simi_vec[vec_ind].second;
              if(simi_s_data[i*num+index]<0) break;
           }
           CHECK(index>=0 && index<simi_s.num());
           if(simi_s_data[i*num+index]<0){
              simi_s_data[i*num+index] = 3;
              simi_s_data[index*num+i] = 3;
           }

           //select semi dissimilar
          while(true){
            vec_ind = length*2 + caffe_rng_rand()%length;
            index = simi_vec[vec_ind].second;
            if(simi_s_data[i*num+index]<0) break;
          }
          if(simi_s_data[i*num+index]<0){
            simi_s_data[i*num+index] = 2;
            simi_s_data[index*num+i] = 2;
          }

          //select triplet examples
          int posnum = pos_idx_vec.size();
          int negnum = neg_idx_vec.size();
          if(posnum<=0) continue;
          CHECK_GT(bottom_label[i],Dtype(-1.0));
          for(int vi=0;vi<posnum;vi++){
            vec_anchor.push_back(i);
            int pos_idx = pos_idx_vec[vi];
            CHECK_EQ(bottom_label[i],bottom_label[pos_idx]);
            vec_pos.push_back(pos_idx);
            int neg_idx = neg_idx_vec[caffe_rng_rand()%negnum];
            CHECK(bottom_label[i]!=bottom_label[neg_idx]);
            vec_neg.push_back(neg_idx);
          }
        }

        // Calculate constrastive loss
        Dtype margin = this->layer_param_.semicontrast_loss_param().margin();
        Dtype lambda = this->layer_param_.semicontrast_loss_param().lambda();
        Dtype weight = this->layer_param_.semicontrast_loss_param().weight();
        Dtype w_rank = this->layer_param_.semicontrast_loss_param().weight_rank();
        Dtype loss(0.0);
        int num_simi=0,num_dissimi=0;
        for(int i=0;i<num;i++){
           for(int j=0;j<num;j++){
              if (i==j) continue;
              CHECK_EQ(simi_s_data[i*num+j],simi_s_data[j*num+i]);
              Dtype alpha = simi_s_data[i*num+j]>=Dtype(2.0) ? lambda:Dtype(1.0);
              if(simi_s_data[i*num+j]==Dtype(1.0)||simi_s_data[i*num+j]==Dtype(3.0)){
                 loss += alpha * dist_sq_data[i*num+j];
                 num_simi++;
              }
              else if(simi_s_data[i*num+j]==Dtype(0.0)||simi_s_data[i*num+j]==Dtype(2.0)){
                 Dtype dist_d = std::max(static_cast<Dtype>(margin - sqrt(dist_sq_data[i*num+j])),Dtype(0.0));
                 loss += alpha * dist_d*dist_d;
                 num_dissimi++;
              }
           }
        }
        loss = weight * loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
        
        //Calculate ranking loss
        Dtype rloss(0.0);
        Dtype m_rank = this->layer_param_.semicontrast_loss_param().margin_rank();
        CHECK(vec_anchor.size()>0);
        for(int i=vec_anchor.size()-1;i>=0;i--){
          int anc_idx = vec_anchor[i];
          CHECK(bottom_label[anc_idx]>Dtype(-1.0));
          int pos_idx = vec_pos[i];
          CHECK(bottom_label[anc_idx]==bottom_label[pos_idx]);
          int neg_idx = vec_neg[i];
          CHECK(bottom_label[anc_idx]!=bottom_label[neg_idx]);
          rloss+=std::max(m_rank+dist_sq_data[anc_idx*num+pos_idx]-dist_sq_data[anc_idx*num+neg_idx],Dtype(0.0));
        }
        rloss = w_rank * rloss / static_cast<Dtype>(vec_anchor.size()) / Dtype(2.0); 
        //printf("num_simi:%d, num_dissimi:%d, closs:%f, rloss:%f --",num_simi,num_dissimi,loss,rloss);
        printf("---------------------------------------------------- num_simi:%d, closs:%f, rsize:%d rloss:%f --",num_simi,loss,static_cast<int>(vec_anchor.size()),rloss);
        top[0]->mutable_cpu_data()[0] = loss + rloss;
        printf("--ave loss: %f\n",top[0]->cpu_data()[0]);
        //CHECK_EQ(1,0);
     }


   template <typename Dtype>
     void SemiContrastLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        printf("Backward_cpu... ");
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* simi_s_data = simi_s.cpu_data();
        const Dtype* dist_sq_data = dist_sq_.cpu_data();

        int count = bottom[0]->count();
        int num = bottom[0]->num();
        int dim = count/num;

        Dtype margin = this->layer_param_.semicontrast_loss_param().margin();
        Dtype lambda = this->layer_param_.semicontrast_loss_param().lambda();
        Dtype weight = this->layer_param_.semicontrast_loss_param().weight();

        Dtype alpha = 2 * weight* top[0]->cpu_diff()[0] / static_cast<Dtype>(num);
        caffe_set(count,Dtype(0.0),bottom_diff);
        //calculate contrastive gradient
        for(int i=0;i<num;i++){
           for(int j=0;j<num;j++){
              if (i == j) continue;
              CHECK_EQ(simi_s_data[i*num+j],simi_s_data[j*num+i]);
              Dtype gama = (simi_s_data[i*num+j]>=Dtype(2.0)) ? lambda*alpha:alpha;
              if(simi_s_data[i*num+j]==Dtype(1.0)||simi_s_data[i*num+j]==Dtype(3.0)){
                 caffe_sub(dim,bottom_data+i*dim,bottom_data+j*dim,data_diff_.mutable_cpu_data());
                 caffe_cpu_axpby(dim,gama,data_diff_.cpu_data(),Dtype(1.0),bottom_diff+i*dim);
              }
              else if(simi_s_data[i*num+j]==Dtype(0.0)||simi_s_data[i*num+j]==Dtype(2.0)){
                 Dtype dist = sqrt(dist_sq_data[i*num+j]);
                 Dtype mdist = margin - dist;
                 Dtype beta = -gama * mdist / (dist + Dtype(1e-4));
                 if(mdist>Dtype(0.0)){
                    caffe_sub(dim,bottom_data+i*dim,bottom_data+j*dim,data_diff_.mutable_cpu_data());
                    caffe_cpu_axpby(dim,beta,data_diff_.cpu_data(),Dtype(1.0),bottom_diff+i*dim);
                 }
              }
           }
        }
        //calculate triplet ranking gradient
        Dtype m_rank = this->layer_param_.semicontrast_loss_param().margin_rank();
        Dtype w_rank = this->layer_param_.semicontrast_loss_param().weight_rank();
        if(static_cast<int>(vec_anchor.size())>0)
           alpha = w_rank * top[0]->cpu_diff()[0] / static_cast<Dtype>(vec_anchor.size());
        else
           alpha = Dtype(0.0);
        int num_count = 0;
        for(int i = vec_anchor.size()-1;i>=0;i--){
          int anc_idx = vec_anchor[i];
          int pos_idx = vec_pos[i];
          int neg_idx = vec_neg[i];
          if(dist_sq_data[anc_idx*num+neg_idx]-dist_sq_data[anc_idx*num+pos_idx]-m_rank<Dtype(0.0)){
            num_count++;
            //for the anchor example --1
            caffe_sub(dim,bottom_data+anc_idx*dim,bottom_data+pos_idx*dim,data_diff_.mutable_cpu_data());
            caffe_cpu_axpby(dim, alpha,data_diff_.cpu_data(),
              Dtype(1.0),bottom_diff+anc_idx*dim);
            //for the positive example
            caffe_cpu_axpby(dim,-alpha,data_diff_.cpu_data(),
              Dtype(1.0),bottom_diff+pos_idx*dim);
            //for the anchor example --2
            caffe_sub(dim,bottom_data+anc_idx*dim,bottom_data+neg_idx*dim,data_diff_.mutable_cpu_data());
            caffe_cpu_axpby(dim,-alpha,data_diff_.cpu_data(),
              Dtype(1.0),bottom_diff+anc_idx*dim);
            //for the negative example
            caffe_cpu_axpby(dim, alpha,data_diff_.cpu_data(),
              Dtype(1.0),bottom_diff+neg_idx*dim);
          }
        }
        printf("rsize:%d,num_count:%d, diff:%f %f %f %f\n",static_cast<int>(vec_anchor.size()),num_count,bottom_diff[0],bottom_diff[1],bottom_diff[dim-1],bottom_diff[count-1]);
     }

#ifdef CPU_ONLY
   STUB_GPU(SemiContrastLossLayer);
#endif

   INSTANTIATE_CLASS(SemiContrastLossLayer);
   REGISTER_LAYER_CLASS(SemiContrastLoss);

}  // namespace caffe
