#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <typeinfo>
#include <cmath>
#include <algorithm>
#include <queue>
#include <tuple>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <chrono>
namespace py = pybind11;


// define the object of each pixel position on range image
class PixelCoord {

public:
  int row=0;
  int col=0;
  PixelCoord(int row_, int col_) : row(row_), col(col_) {}

  PixelCoord operator+(const PixelCoord& other) const {
    return PixelCoord(row + other.row, col + other.col);
  }

};



class Depth_Cluster{
    //default is private
    public:
    
    double angle_threshold = 0.0; // angle threshold of two points
    int search_step = 5;  // consider for holes when searching for nearest point
    //int ww = 64;
    //int hh = 512;
    //int width = 64;
    //int height = 512;

    
    static const int packet_w = 256;
    //static const int packet_w = 236;
    //static const int packet_w = 1024;
    static const int dbot_packet_w = 1024;

    static const int width=64;
    static const int dbot_width=16;

    //static const int packet_size=8192; //64 * 128
    //static const int packet_size=16384; //64 * 246
    static const int NEIGH_SIZE=4;    
    static const int height=2*packet_w;
    static const int dbot_height=2*dbot_packet_w;
    //static const int height=256;
    static const int packet_size=width*height;
    static const int dbot_packet_size=dbot_width*dbot_height;
    // this is how to construct the range image, from -3 degree to 25 degree in vertical direction, 64 is the image vertical resolution, 2048 is the horizontal resolution
    const double angle_resolution_x=28.0/64.0/180.0*3.14159;
    const double angle_resolution_y=3.14159*2/2048.0;
    // this defines each single step to search the nearest point
    std::array<PixelCoord, NEIGH_SIZE> Neighborhood={PixelCoord(-1,0),PixelCoord(1,0),PixelCoord(0,-1),PixelCoord(0,1)};


    Depth_Cluster(double input_thresh, int search_step):angle_threshold(input_thresh),search_step(search_step){}
    
    std::array<int, packet_size> Assign_label_one(std::array<int, packet_size> label_array, double *range_img, int x_location, int y_location){

    	int x_upper_bound=this->width-0;
    	int x_lower_bound=0;
    	int y_upper_bound=this->height-0;
    	int y_lower_bound=0;
        int s_step=this->search_step;
        float a_threshold=this->angle_threshold;
        float dist_th_x=0.2;
        float dist_th_y=1.0;

        double current_indicator=0.0;
        double relative_dist=0.0;
        double d_1=0.0;
        double d_2=0.0;

        std::queue<PixelCoord> labeling_queue;
        int start_lable=label_array[y_upper_bound*x_location+y_location];
   

        PixelCoord start_pixel=PixelCoord(x_location,y_location);
        labeling_queue.push(start_pixel);
        while (!labeling_queue.empty()) {
            const PixelCoord current = labeling_queue.front();
            labeling_queue.pop();

            PixelCoord move_x_down = current+Neighborhood[0];
            double count_temp=1;
            while (range_img[move_x_down.row*y_upper_bound+move_x_down.col]<0.001f && move_x_down.row > x_lower_bound) {
                move_x_down=move_x_down+Neighborhood[0];
                count_temp+=1;
                if (count_temp>s_step) break;
            }
            if (move_x_down.row > x_lower_bound && label_array[y_upper_bound*move_x_down.row+move_x_down.col]==0 && range_img[move_x_down.row*y_upper_bound+move_x_down.col]>0.001f){
            //if (move_x_down.row > x_lower_bound && label_array[y_upper_bound*move_x_down.row+move_x_down.col]==0){    
                d_1=std::max(range_img[move_x_down.row*y_upper_bound+move_x_down.col],range_img[current.row*y_upper_bound+current.col]);
                d_2=std::min(range_img[move_x_down.row*y_upper_bound+move_x_down.col],range_img[current.row*y_upper_bound+current.col]);
                current_indicator=std::atan(std::sin(count_temp*angle_resolution_x)*d_2/(d_1-d_2*std::cos(count_temp*angle_resolution_x)));
                //relative_dist=std::sqrt(std::pow(d_1,2)+std::pow(d_2,2)-2*d_1*d_2*std::cos(angle_resolution_x));
                if (current_indicator>a_threshold && relative_dist < dist_th_x) {
                    labeling_queue.push(move_x_down);
                    label_array[y_upper_bound*move_x_down.row+move_x_down.col]=start_lable;
                }

            }



            PixelCoord move_x_up = current+Neighborhood[1];
            count_temp=1;
            while (range_img[move_x_up.row*y_upper_bound+move_x_up.col]<0.001f && move_x_up.row < x_upper_bound) {
                move_x_up=move_x_up+Neighborhood[1];
                count_temp+=1;
                if (count_temp>s_step) break;
            }

            if (move_x_up.row < x_upper_bound && label_array[y_upper_bound*move_x_up.row+move_x_up.col]==0 && range_img[move_x_up.row*y_upper_bound+move_x_up.col]>0.001f){
                d_1=std::max(range_img[move_x_up.row*y_upper_bound+move_x_up.col],range_img[current.row*y_upper_bound+current.col]);
                d_2=std::min(range_img[move_x_up.row*y_upper_bound+move_x_up.col],range_img[current.row*y_upper_bound+current.col]);
                current_indicator=std::atan(std::sin(count_temp*angle_resolution_x)*d_2/(d_1-d_2*std::cos(count_temp*angle_resolution_x)));
                //relative_dist=std::sqrt(std::pow(d_1,2)+std::pow(d_2,2)-2*d_1*d_2*std::cos(angle_resolution_x));
                if (current_indicator>a_threshold && relative_dist < dist_th_x) {
                    labeling_queue.push(move_x_up);
                    label_array[y_upper_bound*move_x_up.row+move_x_up.col]=start_lable;
                }

            }

            

            PixelCoord move_y_down = current+Neighborhood[2];
            count_temp=1;
            while (range_img[move_y_down.row*y_upper_bound+move_y_down.col]<0.001f && move_y_down.col > y_lower_bound) {
                move_y_down=move_y_down+Neighborhood[2];
                count_temp+=1;
                if (count_temp>s_step) break;
            }

            if (move_y_down.col > y_lower_bound && label_array[y_upper_bound*move_y_down.row+move_y_down.col]==0 && range_img[move_y_down.row*y_upper_bound+move_y_down.col]>0.001f){
                d_1=std::max(range_img[move_y_down.row*y_upper_bound+move_y_down.col],range_img[current.row*y_upper_bound+current.col]);
                d_2=std::min(range_img[move_y_down.row*y_upper_bound+move_y_down.col],range_img[current.row*y_upper_bound+current.col]);
                current_indicator=std::atan(std::sin(count_temp*angle_resolution_y)*d_2/(d_1-d_2*std::cos(count_temp*angle_resolution_y)));
                //relative_dist=std::sqrt(std::pow(d_1,2)+std::pow(d_2,2)-2*d_1*d_2*std::cos(angle_resolution_y));
                if (current_indicator>a_threshold && relative_dist < dist_th_y) {
                    labeling_queue.push(move_y_down);
                    label_array[y_upper_bound*move_y_down.row+move_y_down.col]=start_lable;
                }

            }

            PixelCoord move_y_up = current+Neighborhood[3];
            count_temp=1;
            while (range_img[move_y_up.row*y_upper_bound+move_y_up.col]<0.001f && move_y_up.col < y_upper_bound) {
                move_y_up=move_y_up+Neighborhood[3];
                count_temp+=1;
                if (count_temp>s_step) break;
            }

            if (move_y_up.col < y_upper_bound && label_array[y_upper_bound*move_y_up.row+move_y_up.col]==0 && range_img[move_y_up.row*y_upper_bound+move_y_up.col]>0.001f){
                d_1=std::max(range_img[move_y_up.row*y_upper_bound+move_y_up.col],range_img[current.row*y_upper_bound+current.col]);
                d_2=std::min(range_img[move_y_up.row*y_upper_bound+move_y_up.col],range_img[current.row*y_upper_bound+current.col]);
                current_indicator=std::atan(std::sin(count_temp*angle_resolution_y)*d_2/(d_1-d_2*std::cos(count_temp*angle_resolution_y)));
                //relative_dist=std::sqrt(std::pow(d_1,2)+std::pow(d_2,2)-2*d_1*d_2*std::cos(angle_resolution_y));
                if (current_indicator>a_threshold && relative_dist < dist_th_y) {
                    labeling_queue.push(move_y_up);
                    label_array[y_upper_bound*move_y_up.row+move_y_up.col]=start_lable;
                }

            }
        }
        return label_array;
    }


    std::array<int, packet_size>  Depth_cluster(py::array_t<double> input_array){

		auto buf1 = input_array.request();
		double *ptr1 = (double *) buf1.ptr;
		assert (width*height== buf1.shape[0]);

        std::array<int, packet_size> label_instance=std::array<int, packet_size>();
		int current_lable=0;
		for (size_t idx = 0; idx < this->width; idx+=1){
            for (size_t idy = 0; idy < this->height; idy+=1){ // 4
                
				if (ptr1[height*idx+idy]<0.001f){ 
                    label_instance[height*idx+idy]=0;
					continue;
				} 

                if (label_instance[height*idx+idy]<current_lable &&label_instance[height*idx+idy]>0) continue;
				
				current_lable+=1;
				label_instance[height*idx+idy]=current_lable;
                label_instance=Assign_label_one(label_instance, ptr1, idx, idy);
			}
		}

		return label_instance;
    }

std::array<int, dbot_packet_size> dbot_Assign_label_one(std::array<int, dbot_packet_size> label_array, double *range_img, int x_location, int y_location){

    	int x_upper_bound=this->dbot_width-0;
    	int x_lower_bound=0;
    	int y_upper_bound=this->dbot_height-0;
    	int y_lower_bound=0;
        int s_step=this->search_step;
        float a_threshold=this->angle_threshold;
        float dist_th_x=0.2;
        float dist_th_y=1.0;

        double current_indicator=0.0;
        double relative_dist=0.0;
        double d_1=0.0;
        double d_2=0.0;

        std::queue<PixelCoord> labeling_queue;
        int start_lable=label_array[y_upper_bound*x_location+y_location];
   

        PixelCoord start_pixel=PixelCoord(x_location,y_location);
        labeling_queue.push(start_pixel);
        while (!labeling_queue.empty()) {
            const PixelCoord current = labeling_queue.front();
            labeling_queue.pop();

            PixelCoord move_x_down = current+Neighborhood[0];
            double count_temp=1;
            while (range_img[move_x_down.row*y_upper_bound+move_x_down.col]<0.001f && move_x_down.row > x_lower_bound) {
                move_x_down=move_x_down+Neighborhood[0];
                count_temp+=1;
                if (count_temp>s_step) break;
            }
            if (move_x_down.row > x_lower_bound && label_array[y_upper_bound*move_x_down.row+move_x_down.col]==0 && range_img[move_x_down.row*y_upper_bound+move_x_down.col]>0.001f){
            //if (move_x_down.row > x_lower_bound && label_array[y_upper_bound*move_x_down.row+move_x_down.col]==0){    
                d_1=std::max(range_img[move_x_down.row*y_upper_bound+move_x_down.col],range_img[current.row*y_upper_bound+current.col]);
                d_2=std::min(range_img[move_x_down.row*y_upper_bound+move_x_down.col],range_img[current.row*y_upper_bound+current.col]);
                current_indicator=std::atan(std::sin(count_temp*angle_resolution_x)*d_2/(d_1-d_2*std::cos(count_temp*angle_resolution_x)));
                //relative_dist=std::sqrt(std::pow(d_1,2)+std::pow(d_2,2)-2*d_1*d_2*std::cos(angle_resolution_x));
                if (current_indicator>a_threshold && relative_dist < dist_th_x) {
                    labeling_queue.push(move_x_down);
                    label_array[y_upper_bound*move_x_down.row+move_x_down.col]=start_lable;
                }

            }



            PixelCoord move_x_up = current+Neighborhood[1];
            count_temp=1;
            while (range_img[move_x_up.row*y_upper_bound+move_x_up.col]<0.001f && move_x_up.row < x_upper_bound) {
                move_x_up=move_x_up+Neighborhood[1];
                count_temp+=1;
                if (count_temp>s_step) break;
            }

            if (move_x_up.row < x_upper_bound && label_array[y_upper_bound*move_x_up.row+move_x_up.col]==0 && range_img[move_x_up.row*y_upper_bound+move_x_up.col]>0.001f){
                d_1=std::max(range_img[move_x_up.row*y_upper_bound+move_x_up.col],range_img[current.row*y_upper_bound+current.col]);
                d_2=std::min(range_img[move_x_up.row*y_upper_bound+move_x_up.col],range_img[current.row*y_upper_bound+current.col]);
                current_indicator=std::atan(std::sin(count_temp*angle_resolution_x)*d_2/(d_1-d_2*std::cos(count_temp*angle_resolution_x)));
                //relative_dist=std::sqrt(std::pow(d_1,2)+std::pow(d_2,2)-2*d_1*d_2*std::cos(angle_resolution_x));
                if (current_indicator>a_threshold && relative_dist < dist_th_x) {
                    labeling_queue.push(move_x_up);
                    label_array[y_upper_bound*move_x_up.row+move_x_up.col]=start_lable;
                }

            }

            

            PixelCoord move_y_down = current+Neighborhood[2];
            count_temp=1;
            while (range_img[move_y_down.row*y_upper_bound+move_y_down.col]<0.001f && move_y_down.col > y_lower_bound) {
                move_y_down=move_y_down+Neighborhood[2];
                count_temp+=1;
                if (count_temp>s_step) break;
            }

            if (move_y_down.col > y_lower_bound && label_array[y_upper_bound*move_y_down.row+move_y_down.col]==0 && range_img[move_y_down.row*y_upper_bound+move_y_down.col]>0.001f){
                d_1=std::max(range_img[move_y_down.row*y_upper_bound+move_y_down.col],range_img[current.row*y_upper_bound+current.col]);
                d_2=std::min(range_img[move_y_down.row*y_upper_bound+move_y_down.col],range_img[current.row*y_upper_bound+current.col]);
                current_indicator=std::atan(std::sin(count_temp*angle_resolution_y)*d_2/(d_1-d_2*std::cos(count_temp*angle_resolution_y)));
                //relative_dist=std::sqrt(std::pow(d_1,2)+std::pow(d_2,2)-2*d_1*d_2*std::cos(angle_resolution_y));
                if (current_indicator>a_threshold && relative_dist < dist_th_y) {
                    labeling_queue.push(move_y_down);
                    label_array[y_upper_bound*move_y_down.row+move_y_down.col]=start_lable;
                }

            }

            PixelCoord move_y_up = current+Neighborhood[3];
            count_temp=1;
            while (range_img[move_y_up.row*y_upper_bound+move_y_up.col]<0.001f && move_y_up.col < y_upper_bound) {
                move_y_up=move_y_up+Neighborhood[3];
                count_temp+=1;
                if (count_temp>s_step) break;
            }

            if (move_y_up.col < y_upper_bound && label_array[y_upper_bound*move_y_up.row+move_y_up.col]==0 && range_img[move_y_up.row*y_upper_bound+move_y_up.col]>0.001f){
                d_1=std::max(range_img[move_y_up.row*y_upper_bound+move_y_up.col],range_img[current.row*y_upper_bound+current.col]);
                d_2=std::min(range_img[move_y_up.row*y_upper_bound+move_y_up.col],range_img[current.row*y_upper_bound+current.col]);
                current_indicator=std::atan(std::sin(count_temp*angle_resolution_y)*d_2/(d_1-d_2*std::cos(count_temp*angle_resolution_y)));
                //relative_dist=std::sqrt(std::pow(d_1,2)+std::pow(d_2,2)-2*d_1*d_2*std::cos(angle_resolution_y));
                if (current_indicator>a_threshold && relative_dist < dist_th_y) {
                    labeling_queue.push(move_y_up);
                    label_array[y_upper_bound*move_y_up.row+move_y_up.col]=start_lable;
                }

            }
        }
        return label_array;
    }


    std::array<int, dbot_packet_size>  dbot_Depth_cluster(py::array_t<double> input_array){

		auto buf1 = input_array.request();
		double *ptr1 = (double *) buf1.ptr;
		assert (dbot_width*dbot_height== buf1.shape[0]);

        std::array<int, dbot_packet_size> label_instance=std::array<int, dbot_packet_size>();
		int current_lable=0;
		for (size_t idx = 0; idx < this->dbot_width; idx+=1){
            for (size_t idy = 0; idy < this->dbot_height; idy+=1){ // 4
                
				if (ptr1[dbot_height*idx+idy]<0.001f){ 
                    label_instance[dbot_height*idx+idy]=0;
					continue;
				} 

                if (label_instance[dbot_height*idx+idy]<current_lable &&label_instance[dbot_height*idx+idy]>0) continue;
				
				current_lable+=1;
				label_instance[dbot_height*idx+idy]=current_lable;
                label_instance=dbot_Assign_label_one(label_instance, ptr1, idx, idy);
			}
		}

		return label_instance;
    }


    std::array<int, packet_size> Assign_label_packet(std::array<int, packet_size> label_array, double *range_img, int x_location, int y_location){

    	int x_upper_bound=this->width;
    	int x_lower_bound=0;
    	int y_upper_bound=this->height;
    	int y_lower_bound=0;
        double current_indicator=0.0;
        double d_1=0.0;
        double d_2=0.0;

        std::queue<PixelCoord> labeling_queue;
        int start_lable=label_array[this->height*x_location+y_location];
   

        PixelCoord start_pixel=PixelCoord(x_location,y_location);
        labeling_queue.push(start_pixel);
        //#pragma omp parallel while
        while (!labeling_queue.empty()) {
            const PixelCoord current = labeling_queue.front();
            labeling_queue.pop();

            PixelCoord move_x_down = current+Neighborhood[0];
            double count_temp=1;
            while (range_img[move_x_down.row*this->height+move_x_down.col]<0.001f && move_x_down.row > x_lower_bound) {
                move_x_down=move_x_down+Neighborhood[0];
                count_temp+=1;
                if (count_temp>this->search_step) break;
            }
            if (move_x_down.row > x_lower_bound && label_array[this->height*move_x_down.row+move_x_down.col]==0 && range_img[move_x_down.row*this->height+move_x_down.col]>0.001f){
            //if (move_x_down.row > x_lower_bound && range_img[move_x_down.row*this->height+move_x_down.col]>0.001f){
                d_1=std::max(range_img[move_x_down.row*this->height+move_x_down.col],range_img[current.row*this->height+current.col]);
                d_2=std::min(range_img[move_x_down.row*this->height+move_x_down.col],range_img[current.row*this->height+current.col]);
                current_indicator=atan(sin(count_temp*angle_resolution_x)*d_2/(d_1-d_2*cos(count_temp*angle_resolution_x)));
                if (current_indicator>this->angle_threshold) {
                    labeling_queue.push(move_x_down);
                    label_array[this->height*move_x_down.row+move_x_down.col]=start_lable;
                }

            }

            PixelCoord move_x_up = current+Neighborhood[1];
            count_temp=1;
            while (range_img[move_x_up.row*this->height+move_x_up.col]<0.001f && move_x_up.row < x_upper_bound) {
                move_x_up=move_x_up+Neighborhood[1];
                count_temp+=1;
                if (count_temp>this->search_step) break;
            }

            if (move_x_up.row < x_upper_bound && label_array[this->height*move_x_up.row+move_x_up.col]==0 && range_img[move_x_up.row*this->height+move_x_up.col]>0.001f){
            //if (move_x_up.row < x_upper_bound && range_img[move_x_up.row*this->height+move_x_up.col]>0.001f){
                d_1=std::max(range_img[move_x_up.row*this->height+move_x_up.col],range_img[current.row*this->height+current.col]);
                d_2=std::min(range_img[move_x_up.row*this->height+move_x_up.col],range_img[current.row*this->height+current.col]);
                current_indicator=atan(sin(count_temp*angle_resolution_x)*d_2/(d_1-d_2*cos(count_temp*angle_resolution_x)));
                if (current_indicator>this->angle_threshold) {
                    labeling_queue.push(move_x_up);
                    label_array[this->height*move_x_up.row+move_x_up.col]=start_lable;
                }

            }

            PixelCoord move_y_down = current+Neighborhood[2];
            count_temp=1;
            while (range_img[move_y_down.row*this->height+move_y_down.col]<0.001f && move_y_down.col > y_lower_bound) {
                move_y_down=move_y_down+Neighborhood[2];
                count_temp+=1;
                if (count_temp>this->search_step) break;
            }

            if (move_y_down.col > y_lower_bound && label_array[this->height*move_y_down.row+move_y_down.col]==0 && range_img[move_y_down.row*this->height+move_y_down.col]>0.001f){
            //if (move_y_down.col > y_lower_bound && range_img[move_y_down.row*this->height+move_y_down.col]>0.001f){
                d_1=std::max(range_img[move_y_down.row*this->height+move_y_down.col],range_img[current.row*this->height+current.col]);
                d_2=std::min(range_img[move_y_down.row*this->height+move_y_down.col],range_img[current.row*this->height+current.col]);
                current_indicator=atan(sin(count_temp*angle_resolution_y)*d_2/(d_1-d_2*cos(count_temp*angle_resolution_y)));
                if (current_indicator>this->angle_threshold) {
                    labeling_queue.push(move_y_down);
                    label_array[this->height*move_y_down.row+move_y_down.col]=start_lable;
                }

            }

            PixelCoord move_y_up = current+Neighborhood[3];
            count_temp=1;
            while (range_img[move_y_up.row*this->height+move_y_up.col]<0.001f && move_y_up.col < y_upper_bound) {
                move_y_up=move_y_up+Neighborhood[3];
                count_temp+=1;
                if (count_temp>this->search_step) break;
            }

            if (move_y_up.col < y_upper_bound && label_array[this->height*move_y_up.row+move_y_up.col]==0 && range_img[move_y_up.row*this->height+move_y_up.col]>0.001f){
            //if (move_y_up.col < y_upper_bound && range_img[move_y_up.row*this->height+move_y_up.col]>0.001f){
                d_1=std::max(range_img[move_y_up.row*this->height+move_y_up.col],range_img[current.row*this->height+current.col]);
                d_2=std::min(range_img[move_y_up.row*this->height+move_y_up.col],range_img[current.row*this->height+current.col]);
                current_indicator=atan(sin(count_temp*angle_resolution_y)*d_2/(d_1-d_2*cos(count_temp*angle_resolution_y)));
                if (current_indicator>this->angle_threshold) {
                    labeling_queue.push(move_y_up);
                    label_array[this->height*move_y_up.row+move_y_up.col]=start_lable;
                }

            }
        }
        return label_array;
    }


    std::array<int, packet_size>  Packet_cluster(py::array_t<double> input_array, int currentlabel){
    	// use np.reshape(-1) to strip the 2D image to 1d array

		auto buf1 = input_array.request();

		double *ptr1 = (double *) buf1.ptr;

		assert (width*height== buf1.shape[0]);

        // initialize label array
        std::array<int, packet_size> label_instance=std::array<int, packet_size>();

        int current_lable=currentlabel;

        for (size_t idx = 0; idx < this->width; idx+=1){
            for (size_t idy = (int)this->height/2-2; idy < this->height; idy+=4){
                
                // if depth value is almost zero, assign label 0
				if (ptr1[height*idx+idy]<0.001f){ 
                    label_instance[height*idx+idy]=0;
					continue;
				}

                // if label has been assigned before. Here may indicates a bug, the label image is not initialized which means the initial value is random. But I think this risk is ok as the failure probabilty is really small, and we can tolerate even if the failure happens
				if (label_instance[height*idx+idy]<current_lable &&label_instance[height*idx+idy]>0) continue;
                
				current_lable+=1;
				label_instance[height*idx+idy]=current_lable;
                label_instance=Assign_label_packet(label_instance, ptr1, idx, idy);
			}
        }
		return label_instance;
    }
};


PYBIND11_MODULE(Depth_Cluster, m) {
    py::class_<Depth_Cluster>(m, "Depth_Cluster")
    	.def(py::init<float, int>())
        .def("Assign_label_one", &Depth_Cluster::Assign_label_one)
        .def("dbot_Assign_label_one", &Depth_Cluster::dbot_Assign_label_one)
        .def("Assign_label_packet", &Depth_Cluster::Assign_label_packet)
        .def("Depth_cluster", &Depth_Cluster::Depth_cluster)
        .def("dbot_Depth_cluster", &Depth_Cluster::dbot_Depth_cluster)
        .def("Packet_cluster", &Depth_Cluster::Packet_cluster);
}
