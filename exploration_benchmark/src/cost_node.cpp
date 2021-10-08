/******************************************************************************
Copyright (c) 2021 Tsinghua University
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
Author: Yuanfan Xu (xuyf20@mails.tsinghua.edu.cn)
*******************************************************************************/

// --------------------- ros things
#include "ros/ros.h"
#include "move_base_msgs/MoveBaseAction.h"
#include "move_base_msgs/MoveBaseGoal.h"
#include "visualization_msgs/Marker.h"
#include <tf/transform_listener.h>
#include "nav_msgs/OccupancyGrid.h"
#include <actionlib/client/simple_action_client.h>
#include "cartographer_ros_msgs/TrajectoryQuery.h"
#include <mutex>

// --------------------- MMPF include things
#include<fstream>
#include<sstream>
#include<iostream>
#include<iomanip>
#include<string>
#include<cstdlib>
#define RESOLUTION 0.05
#define K_ATTRACT 1
#define ETA_REPLUSIVE 3
#define DIS_OBTSTACLE 6
#define DISTANCE_THRES_VALID_OBSTACLE 160
#define THRESHOLD_TRANSFORM 0.5
#define ROBOT_INTERFERE_RADIUS 50
#define LARGEST_MAP_DISTANCE 500000 // 500*1000

// #define FILE_DEBUG
// --------------------- ROS global variable
nav_msgs::OccupancyGrid     mapData, costmapData;
std::mutex _mt;
geometry_msgs::PointStamped clickedpoint;
visualization_msgs::Marker  points,line;

// -----------------------------------  Subscribers callback functions-
void mapCallBack(const nav_msgs::OccupancyGrid::ConstPtr& msg)
{
    //_mt.lock();
	mapData=*msg;
    // std::cout << "assigner receives map" << std::endl;
    //_mt.unlock();
}

void costmapMergedCallBack(const nav_msgs::OccupancyGrid::ConstPtr& msg)
{
    //_mt.lock();
	costmapData=*msg;
    // std::cout << "assigner receives costmap" << std::endl;
    //_mt.unlock();
}

geometry_msgs::Point p;  
void rvizCallBack(const geometry_msgs::PointStamped::ConstPtr& msg)
{ 
	p.x=msg->point.x;
	p.y=msg->point.y;
	p.z=msg->point.z;
	points.points.push_back(p);
}


// --------------------------------------------------//
// ----------------------Main------------------------//
// --------------------------------------------------//
int main(int argc, char** argv) {
    // ---------------------------------------- ros initialization;
    std::string map_topic, costmap_topic, trajectory_query_name, output_file, output_map_file;
    std::string robot_frame, robot_base_frame;
    int rateHz;
    ros::init(argc, argv, "mmpf_node");
    ros::NodeHandle nh;
	std::string nodename, ns, robot_ano_frame_suffix, robot_ano_frame_preffix;
    int rotation_count   = 0, n_robot, this_robot_idx;
    float inflation_radius;    // max 4 degree;
    bool start_condition = true;
    int no_targets_count = 0;

    float rotation_w[3]  = {0.866,  0.500, 1.0};
    float rotation_z[3]  = {0.5  , -0.866, 0.0};
     
    nodename=ros::this_node::getName();

    ros::param::param<std::string>(nodename+"/map_topic", map_topic, "robot1/map"); 
    ros::param::param<std::string>(nodename+"/costmap_topic", costmap_topic, "robot1/move_base/global_costmap/costmap");
    ros::param::param<std::string>(nodename+"/robot_base_frame", robot_base_frame, "robot1/base_footprint");
    ros::param::param<std::string>(nodename+"/robot_frame", robot_frame, "robot1/map");
    ros::param::param<std::string>(nodename+"/namespace", ns, "robot1");
    ros::param::param<int>(nodename+"/rate", rateHz, 1);
    ros::param::param<float>(nodename+"/inflation_radius", inflation_radius, 6);
    ros::param::param<int>(nodename+"/n_robot", n_robot, 1);
    ros::param::param<int>(nodename+"/this_robot_idx", this_robot_idx, 1);
    ros::param::param<std::string>(nodename+"/robot_ano_frame_preffix", robot_ano_frame_preffix, "robot");
    ros::param::param<std::string>(nodename+"/robot_ano_frame_suffix", robot_ano_frame_suffix, "/base_footprint");
    ros::param::param<std::string>(nodename+"/trajectory_query_name", trajectory_query_name, "robot1/trajectory_query");
    ros::param::param<std::string>(nodename+"/output_file", output_file, "/home/nics/catkin_ws/src/SMMR-Explore/src/mmpf/robot1_mmpf_trajectory.txt");
    ros::param::param<std::string>(nodename+"/output_map_file", output_map_file, "/home/nics/catkin_ws/src/SMMR-Explore/src/mmpf/robot1_mmpf_explored_map.txt");

    ros::Rate rate(rateHz);

    // ------------------------------------- subscribe the map topics & clicked points
    ros::Subscriber sub       = nh.subscribe<nav_msgs::OccupancyGrid>(map_topic, 20 ,mapCallBack);
    ros::Subscriber costMapSub= nh.subscribe<nav_msgs::OccupancyGrid>(costmap_topic, 20, costmapMergedCallBack);
    ros::Subscriber rviz_sub  = nh.subscribe<geometry_msgs::PointStamped>("/clicked_point", 10, rvizCallBack);

    // ------------------------------------- subscribe the map topics & clicked points
	tf::TransformListener listener;

    // ------------------------------------- publish the detected points for following processing & display
	ros::Publisher pub          = nh.advertise<visualization_msgs::Marker>(nodename+"_shapes", 100);
	ros::Publisher pub_centroid = nh.advertise<visualization_msgs::Marker>(nodename+"_detected_frontier_centroid", 10);


    // ------------------------------------- wait until map is received
    std::cout << ns << "wait for map "<< std::endl;
	while ( mapData.data.size() < 1 ){  ros::spinOnce();  ros::Duration(0.1).sleep(); }
    std::cout << ns << "wait for costmap "<< std::endl;
    while ( costmapData.data.size()<1)  {  ros::spinOnce();  ros::Duration(0.1).sleep();}
    
    // ------------------------------------- save trajectory when finishing exploration
    ros::ServiceClient trajectory_query_client = nh.serviceClient<cartographer_ros_msgs::TrajectoryQuery>(trajectory_query_name);
    
    std::cout << ns << "wait for trajectory service"<< std::endl;
    trajectory_query_client.waitForExistence();

    // ------------------------------------- action lib    
    actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> ac(ns + "/move_base", true);
    std::cout << ns << "wait for actionserver"<< std::endl;
    ac.waitForServer();
    
    move_base_msgs::MoveBaseGoal robotGoal;
    robotGoal.target_pose.header.frame_id = robot_frame;
    robotGoal.target_pose.pose.position.z = 0;
    robotGoal.target_pose.pose.orientation.z = 1.0;

    // ------------------------------------- initilize the visualized points & lines  
	points.header.frame_id  = mapData.header.frame_id;
	points.header.stamp     = ros::Time(0);
	points.type 			= points.POINTS;
	points.action           = points.ADD;
	points.pose.orientation.w =1.0;
	points.scale.x 			= 0.3; 
	points.scale.y			= 0.3; 
	points.color.r 			= 1.0;   // 255.0/255.0;
	points.color.g 			= 0.0;   // 0.0/255.0;
	points.color.b 			= 0.0;   // 0.0/255.0;
	points.color.a			= 1.0;
	points.lifetime         = ros::Duration();

	line.header.frame_id    = mapData.header.frame_id;
	line.header.stamp       = ros::Time(0);
	line.type				= line.LINE_LIST;
	line.action             = line.ADD;
	line.pose.orientation.w = 1.0;
	line.scale.x 			= 0.03;
	line.scale.y			= 0.03;
	line.color.r			= 1.0;   // 0.0/255.0;
	line.color.g			= 0.0;   // 0.0/255.0;
	line.color.b 			= 1.0;   // 236.0/255.0;
	line.color.a 			= 1.0;
	line.lifetime           = ros::Duration();

    // -------------------------------------Initialize all robots' frame;
    std::string robots_frame[n_robot];

    for (int i = 1; i < n_robot+1; i++){

        std::stringstream ss;              
        ss << robot_ano_frame_preffix;
        ss << i;
        ss << robot_ano_frame_suffix;

        robots_frame[i-1] = ss.str();
    }

    // ------------------------------------- wait the clicked points  
    std::cout << ns << "wait to start" << std::endl;
    while(points.points.size()<1)
	{
		ros::spinOnce();
		pub.publish(points);
	}

    // -------------------------------------clear clicked points
	points.points.clear();
	pub.publish(points);

     
    while(ros::ok()){
        // ---------------------------------------- variables from ROS input;
        int HEIGHT = mapData.info.height;
        int WIDTH  = mapData.info.width;
        
        // ---------------------------------------- define variables;
        std::vector<int* > obstacles, path, targets;
        int currentLoc[2], goal[2]; //target[2], obstacle[2]
        float  minDis2Frontier;
        std::ifstream infile;
        int map[HEIGHT*WIDTH];

        // ---------------------------------------- initialize the map
        for (int i=0; i<HEIGHT; i++)
        {
            for (int j=0; j<WIDTH; j++)
            {
                map[i*WIDTH + j] = (int) mapData.data[i*mapData.info.width + j];
            }
        }

        // ------------------------------------------ find the obstacles & targets
        for (int i = 2; i < HEIGHT-2; i++){
            for (int j = 2; j < WIDTH-2; j++){
                if(map[i*WIDTH + j] == 100){
                    obstacles.push_back(new int[2]{i,j});
                }
                else if(map[i*WIDTH + j] == -1){
                    // accessiable frontiers
                    int numFree = 0, temp1 = 0;

                    if (map[(i + 1)*WIDTH + j] == 0){
                        temp1 += (map[(i + 2)*WIDTH + j    ] == 0) ? 1 : 0;
                        temp1 += (map[(i + 1)*WIDTH + j + 1] == 0) ? 1 : 0;
                        temp1 += (map[(i + 1)*WIDTH + j - 1] == 0) ? 1 : 0;
                        numFree += (temp1 > 0);
                    }

                    if (map[i*WIDTH + j + 1] == 0){
                        temp1 = 0;
                        temp1 += (map[      i*WIDTH + j + 2] == 0) ? 1 : 0;
                        temp1 += (map[(i + 1)*WIDTH + j + 1] == 0) ? 1 : 0;
                        temp1 += (map[(i - 1)*WIDTH + j + 1] == 0) ? 1 : 0;
                        numFree += (temp1 > 0);
                    }

                    if (map[(i - 1) *WIDTH + j] == 0){
                        temp1 = 0;
                        temp1 += (map[(i - 1)*WIDTH + j + 1] == 0) ? 1 : 0;
                        temp1 += (map[(i - 1)*WIDTH + j - 1] == 0) ? 1 : 0;
                        temp1 += (map[(i - 2)*WIDTH + j    ] == 0) ? 1 : 0;
                        numFree += (temp1 > 0);
                    }

                    if (map[i * WIDTH + j - 1] == 0){
                        temp1 = 0;
                        temp1 += (map[    i  *WIDTH + j - 2] == 0) ? 1 : 0;
                        temp1 += (map[(i + 1)*WIDTH + j - 1] == 0) ? 1 : 0;
                        temp1 += (map[(i - 1)*WIDTH + j - 1] == 0) ? 1 : 0;
                        numFree += (temp1 > 0);
                    }

                    if( numFree > 0 ) {
                        targets.push_back(new int[2]{i,j});
                    }
                }
            }
        }
        
         // ------------------------------------------ remove targets within the inflation layer of costmap.
        {
            for (int idx_target = targets.size()-1; idx_target >= 0; idx_target--) {
                
                float loc_x = targets[idx_target][1]*mapData.info.resolution + mapData.info.origin.position.x;
                float loc_y = targets[idx_target][0]*mapData.info.resolution + mapData.info.origin.position.y;
                int index_costmap = (loc_y - costmapData.info.origin.position.y)/costmapData.info.resolution * costmapData.info.width + (loc_x - costmapData.info.origin.position.x)/costmapData.info.resolution;
                if (costmapData.data[index_costmap] >0){
                    targets.erase(targets.begin() + idx_target);
                    continue;
                }
            }
            // std::cout << "(costmap) number targets after erase" << targets.size() << std::endl;
        }
        
        // ------------------------------------------ remove targets within the inflation radius of obstacles.
        {
            for(int idx_target = targets.size()-1; idx_target>=0; idx_target--) {
                for (int i = 0; i < obstacles.size(); i++) {
                    if (abs(targets[idx_target][0] - obstacles[i][0]) +
                        abs(targets[idx_target][1] - obstacles[i][1]) < inflation_radius) {
                        targets.erase(targets.begin() + idx_target);
                        break;
                    }
                }
            }
        }

        // ------------------------------------------ exploration finish detection
        if(targets.size() == 0){
            if(no_targets_count == 4){
                std::cout << "exploration done" << std::endl;
                std::vector<geometry_msgs::PointStamped> path_list;
                cartographer_ros_msgs::TrajectoryQuery srv;
                srv.request.trajectory_id = 0;
                trajectory_query_client.call(srv);
                double trajectory_length, exploration_time;
                exploration_time = srv.response.trajectory[0].header.stamp.sec;
                exploration_time =srv.response.trajectory.back().header.stamp.sec - exploration_time;
                std::cout <<  ns << "exploration_time is:" << exploration_time << " seconds" << std::endl;
                std::ofstream ofile(output_file);
                double trajectory_x = srv.response.trajectory[0].pose.position.x;
                double trajectory_y = srv.response.trajectory[0].pose.position.y;
                                
                ofile << "[";
                ofile << trajectory_x << "," << trajectory_y << std::endl;
                for (int i = 1; i < srv.response.trajectory.size(); i++){
                    double temp_x = srv.response.trajectory[i].pose.position.x;
                    double temp_y = srv.response.trajectory[i].pose.position.y;
                    ofile << temp_x  << ", " <<  temp_y << ";" << std::endl;
                    double delta_x = trajectory_x - temp_x;
                    double delta_y = trajectory_y - temp_y;
                    trajectory_length += sqrt(delta_x*delta_x + delta_y*delta_y);
                    trajectory_x = temp_x;
                    trajectory_y = temp_y; 
                }
                ofile << "]" << std::endl;
                ofile.close();
                std::cout <<  ns << "exploration trajectory length = " << trajectory_length << " meters" << std::endl;
                
                std::ofstream ofile2(output_map_file);
                ofile2 <<  ns << "map Origin (" << mapData.info.origin.position.x << " ," << mapData.info.origin.position.y << ")" << std::endl;
                for(int i = 0; i < mapData.data.size(); i++){
                    ofile2 << mapData.data[i] << " ";
                }
                ofile2.close();

                return 0;
            }   

            no_targets_count ++;
            std::cout << ns << "no targets count = " << no_targets_count << std::endl;
            rate.sleep();
            continue;
        }
        else{
            no_targets_count = 0;
        }

        
        // ---------------------------------------- define the current point;
        tf::StampedTransform  transform;
        int  temp=0;
        while (temp==0){
            try{
                temp=1;
                listener.lookupTransform( mapData.header.frame_id, robot_base_frame, ros::Time(0), transform );
            }
            catch( tf::TransformException ex ){
                temp=0;
                ros::Duration(0.1).sleep();
            }
        }
        currentLoc[0] = floor((transform.getOrigin().y()-mapData.info.origin.position.y)/mapData.info.resolution);
        currentLoc[1] = floor((transform.getOrigin().x()-mapData.info.origin.position.x)/mapData.info.resolution);
        path.push_back( currentLoc );

        // ------------------------------------------ cluster targets into different groups and find the center of each group.
        // Note: x & y value of detected targets are in increasing order because of the detection is in laser scan order.
        std::vector<int* > target_process(targets);
        std::vector<int* > cluster_center;
        std::vector<int>   infoGain_cluster;

        while(target_process.size() > 0){
            std::vector<int* > target_cluster;
            target_cluster.push_back(target_process.back());
            target_process.pop_back();

            bool condition = true;
            while(condition){
                condition = false;
                int size_target_process = target_process.size();
                for (int i = size_target_process-1; i >= 0 ; i--){
                    for (int j = 0; j < target_cluster.size(); j++){
                        int dis_ = abs(target_process[i][0] - target_cluster[j][0]) +  abs(target_process[i][1] - target_cluster[j][1]);
                        if(dis_ < 3){
                            target_cluster.push_back(target_process[i]);
                            target_process.erase(target_process.begin() + i);
                            condition = true;
                            break;
                        }
                    }
                }
            }

            int center_[2]={0, 0};
            int num_ = target_cluster.size();
            for(int i = 0; i < num_; i++){
                center_[0] += target_cluster[i][0];
                center_[1] += target_cluster[i][1];
            }

            float center_float[2] = {float(center_[0]), float(center_[1])};
            center_float[0] = center_float[0]/float(num_);
            center_float[1] = center_float[1]/float(num_);

            float min_dis_ = 100.0;
            int min_idx_   = 10000;
            for(int i = 0; i < num_; i++){
                float temp_dis_ = abs(center_float[0]-float(target_cluster[i][0])) + abs(center_float[1]-float(target_cluster[i][1]));
                if(temp_dis_ < min_dis_){
                    min_dis_ = temp_dis_;
                    min_idx_ = i;
                }
            }

            cluster_center.push_back(new int[2]{target_cluster[min_idx_][0], target_cluster[min_idx_][1]});
            infoGain_cluster.push_back(num_);
        }

        // ------------------------------------------ Display Cluster centroids
        points.points.clear();
        for(int i = 0; i < cluster_center.size(); i++){
            geometry_msgs::Point temp;
            temp.x = cluster_center[i][1] * mapData.info.resolution + mapData.info.origin.position.x;
            temp.y = cluster_center[i][0] * mapData.info.resolution + mapData.info.origin.position.y;
            temp.z = 0;
            points.points.push_back(temp);
        }
        pub_centroid.publish(points); 
        // ------------------------------------------ Calculate the nearest frontier
        int cluster_num = cluster_center.size();
        int min_dis = 1000000;
        int min_idx = -1;
        for(int i = 0; i< cluster_num; i++) {
            int dis = (currentLoc[0] - cluster_center[i][0])*(currentLoc[0] - cluster_center[i][0]) + (currentLoc[1] - cluster_center[i][1])*(currentLoc[1] - cluster_center[i][1]);
            if(dis < min_dis){
                min_dis = dis;
                min_idx = i;
            }
        }

        // goal[0] = path.back()[0];
        // goal[1] = path.back()[1];
      
        if(start_condition){
            tf::StampedTransform  transform;
            int  temp=0;
            while (temp==0){
                try{
                    temp=1;
                    listener.lookupTransform( mapData.header.frame_id, robot_base_frame, ros::Time(0), transform );
                }
                catch( tf::TransformException ex ){
                    temp=0;
                    ros::Duration(0.1).sleep();
                }
            }
            int loc_x = transform.getOrigin().x();
            int loc_y = transform.getOrigin().y();

            robotGoal.target_pose.pose.orientation.z = rotation_z[rotation_count];
            robotGoal.target_pose.pose.orientation.w = rotation_w[rotation_count];
  
            robotGoal.target_pose.pose.position.x = loc_x + 0.2;
            robotGoal.target_pose.pose.position.y = loc_y + 0.2;
        
            start_condition = false;
        }
        else{
            robotGoal.target_pose.pose.orientation.z = 1;
            robotGoal.target_pose.pose.orientation.w = 0;
            robotGoal.target_pose.pose.position.x = cluster_center[min_idx][1]*mapData.info.resolution + mapData.info.origin.position.x;
            robotGoal.target_pose.pose.position.y = cluster_center[min_idx][0]*mapData.info.resolution + mapData.info.origin.position.y;
            // std::cout << robotGoal.target_pose.pose.position.x << std::endl;
            // std::cout << robotGoal.target_pose.pose.position.y << std::endl;
            robotGoal.target_pose.header.stamp    = ros::Time(0);
            ac.sendGoal(robotGoal);
        }
        line.points.clear();

        // ------------------------------------------- keep frequency stable
        // _mt.unlock();
        ros::spinOnce();
        rate.sleep();
    } 

    return 0;

}