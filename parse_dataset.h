#ifndef __PARSE_DATASET_H
#define __PARSE_DATASET_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


class ParseDataset
{
public:
    ParseDataset(const std::string& path) : path_(path) {
        
    }

    void parse() {
        std::string picture_path = get_picture_path();
        std::string groundtruth_path = get_groundtruth_path();
        parse_pictures(picture_path);
        parse_groundtruth(groundtruth_path);
        printf("Find %d picture, %d rects\n", pictures_.size(), bboxes_.size());
    }

    virtual std::string get_picture_path() = 0;
    virtual std::string get_groundtruth_path() = 0;

    void parse_pictures(const std::string& pattern) {
        std::cout << "parse_pictures: " << pattern << std::endl;

        std::vector<cv::String> list;
        cv::glob(pattern, list);
        for (size_t i = 0; i < list.size(); i++) {
            std::string pic = list[i];
            pictures_.push_back(pic);
        }
    }
 
    int num() {
        return (int)pictures_.size();
    }

    cv::Mat read_image(int index, int flag) {
        cv::Mat img;

        if (index < num()) {
            img = cv::imread(pictures_[index], flag);
        }

        return img; 
    }

    cv::Rect get_bbox(int index) {
        if (index < num()) {
            return bboxes_[index];
        } else {
            return cv::Rect(0, 0, 0, 0);
        }
    }

    virtual int parse_groundtruth(const std::string& groundtruth_path) = 0;

    void debug_info() {
        std::cout << "dataset path: " << path_ << std::endl;

        for (size_t i = 0; i < pictures_.size(); i++) {
            std::cout << pictures_[i] << std::endl;
        }

        for (size_t i = 0; i < bboxes_.size(); i++) { 
            cv::Rect bbox = bboxes_[i];
            printf("%d: x=%d, y=%d, width=%d, height=%d\n", i, bbox.x, bbox.y, bbox.width, bbox.height);
        }
    }
    

public:  
    std::string path_;
    std::vector<std::string> pictures_; 
    std::vector<cv::Rect> bboxes_;
};




class ParseDatasetOtb50 : public ParseDataset
{
public:
    ParseDatasetOtb50(const std::string& path) : ParseDataset(path) {}
 

    std::string get_picture_path() {
        std::string picture_path = path_ + "/img/*.jpg";
        return picture_path;
    }

    std::string get_groundtruth_path() {
        std::string groundtruth_path = path_ + "/groundtruth_rect.txt";
        return groundtruth_path;
    }

    int parse_groundtruth(const std::string& groundtruth_path) {
        std::ifstream gt;
        gt.open(groundtruth_path.c_str());
        if (!gt.is_open()) {
            printf("ground truth file can't open: %s\n", groundtruth_path.c_str());
            return -1;
        }

        std::string line;
        std::string x_str, y_str, w_str, h_str;
        while(getline(gt, line)) {
            std::replace(line.begin(), line.end(), ',', ' ');
            std::stringstream ss;
            ss.str(line);
            ss >> x_str >> y_str >> w_str >> h_str;
            int x = atoi(x_str.c_str());
            int y = atoi(y_str.c_str());
            int w = atoi(w_str.c_str());
            int h = atoi(h_str.c_str());

            cv::Rect bbox(x, y, w, h);
            bboxes_.push_back(bbox);
        }

        gt.close();
    }
};


class ParseDatasetVot2013 : public ParseDataset
{
public:
    ParseDatasetVot2013(const std::string& path) : ParseDataset(path) {
        polygons_.clear();
    }

    std::string get_picture_path() {
        std::string picture_path = path_ + "/*.jpg";
        return picture_path;
    }

    std::string get_groundtruth_path() {
        std::string groundtruth_path = path_ + "/groundtruth.txt";
        return groundtruth_path;
    }

    int parse_groundtruth(const std::string& groundtruth_path) {
        std::ifstream gt;
        gt.open(groundtruth_path.c_str());
        if (!gt.is_open()) {
            printf("ground truth file can't open: %s\n", groundtruth_path.c_str());
            return -1;
        }

        std::string line;
        float x1, y1, x2, y2, x3, y3, x4, y4;
        while (getline(gt, line)) {
            std::replace(line.begin(), line.end(), ',', ' ');
            std::stringstream ss;
            ss.str(line);
            ss >> x1 >> y1 >> x2 >> y2 >> x3 >> y3 >> x4 >> y4;
            std::vector<cv::Point2f> polygon;
            polygon.push_back(cv::Point2f(x1, y1));
            polygon.push_back(cv::Point2f(x2, y2));
            polygon.push_back(cv::Point2f(x3, y3));
            polygon.push_back(cv::Point2f(x4, y4));
            cv::Rect_<float> rect = getAxisAlignedBB(polygon);
            cv::Rect bbox((int)rect.x, (int)rect.y, (int)rect.width, (int)rect.height);
            bboxes_.push_back(bbox);
            polygons_.push_back(polygon);
        }
        gt.close();
    }

    cv::Rect_<float> getAxisAlignedBB(std::vector<cv::Point2f> polygon) {
        double cx = double(polygon[0].x + polygon[1].x + polygon[2].x + polygon[3].x) / 4.;
        double cy = double(polygon[0].y + polygon[1].y + polygon[2].y + polygon[3].y) / 4.;
        double x1 = std::min(std::min(std::min(polygon[0].x, polygon[1].x), polygon[2].x), polygon[3].x);
        double x2 = std::max(std::max(std::max(polygon[0].x, polygon[1].x), polygon[2].x), polygon[3].x);
        double y1 = std::min(std::min(std::min(polygon[0].y, polygon[1].y), polygon[2].y), polygon[3].y);
        double y2 = std::max(std::max(std::max(polygon[0].y, polygon[1].y), polygon[2].y), polygon[3].y);
        double A1 = norm(polygon[1] - polygon[2])*norm(polygon[2] - polygon[3]);
        double A2 = (x2 - x1) * (y2 - y1);
        double s = sqrt(A1 / A2);
        double w = s * (x2 - x1) + 1;
        double h = s * (y2 - y1) + 1;
        cv::Rect_<float> rect(cx-1-w/2.0, cy-1-h/2.0, w, h);
        return rect;
    }

    std::vector<cv::Point2f> get_polygon(int index) {
        if (index >= num()) {
            std::vector<cv::Point2f> polygon;
            return polygon;
        } else {
            return polygons_[index];
        }
    }

public:
    std::vector<std::vector<cv::Point2f> > polygons_;    

}; 


class ParseDatasetFactory
{
public:
     ParseDataset* create_parse_dataset(const std::string& path, const std::string& type) {
         if (strcmp("otb50", type.c_str()) == 0) { 
             ParseDatasetOtb50* ptr = new ParseDatasetOtb50(path);
             return (ParseDataset*)ptr;
         }
         else if (strcmp("vot2015", type.c_str()) == 0) {
             ParseDatasetVot2013* ptr = new ParseDatasetVot2013(path);
             return (ParseDataset*)ptr;
         } else {
             return NULL;
         }
     }    

     std::string support_dataset() {
         std::string str = "otb50|vot2015";
         return str;
     }
};






#endif
