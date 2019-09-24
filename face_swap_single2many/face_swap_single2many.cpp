// std
#include <iostream>
#include <fstream>
#include <exception>

// Boost
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/regex.hpp>
#include <boost/timer/timer.hpp>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// face_swap
#include <face_swap/face_swap_engine.h>
#include <face_swap/utilities.h>
#include <face_swap/render_utilities.h>

using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::runtime_error;
using namespace boost::program_options;
using namespace boost::filesystem;

const std::string IMAGE_FILTER =
"(.*\\.(bmp|dib|jpeg|jpg|jpe|jp2|png|pbm|pgm|ppm|sr|ras))";


void getImagesFromDir(const std::string& dir_path, std::vector<std::string>& img_paths)
{
    boost::regex filter(IMAGE_FILTER);
    boost::smatch what;
    directory_iterator end_itr; // Default ctor yields past-the-end
    for (directory_iterator it(dir_path); it != end_itr; ++it)
    {
        // Skip if not a file
        if (!boost::filesystem::is_regular_file(it->status())) continue;

        // Get extension
        std::string ext = it->path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        // Skip if no match
        if (!boost::regex_match(ext, what, filter)) continue;

        img_paths.push_back(it->path().string());
    }
}

void parseInputPaths(const std::vector<std::string>& input_paths, std::vector<std::string>& img_paths, 
	const std::vector<bool>& toggle_input_seg, std::vector<bool>& toggle_img_seg,
	const std::vector<int>& input_max_res, std::vector<int>& img_max_res)
{
	boost::regex filter(IMAGE_FILTER);
	boost::smatch what;

	// For each input path
	for (size_t i = 0; i < input_paths.size(); ++i)
	{
		const string& input_path = input_paths[i];

		// Get extension
		std::string ext = path(input_path).extension().string();
		std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

		if (is_regular_file(input_path) && boost::regex_match(ext, what, filter))
		{
			img_paths.push_back(input_path);
			toggle_img_seg.push_back(toggle_input_seg[i]);
			img_max_res.push_back(input_max_res[i]);
		}
		else if (is_directory(input_path))
		{
			getImagesFromDir(input_path, img_paths);
			toggle_img_seg.resize(img_paths.size(), toggle_input_seg[i]);
			img_max_res.resize(img_paths.size(), input_max_res[i]);
		}
		else throw error("input must be contain paths to image files or input directories!");
	}
}

void logError(std::ofstream& log, const std::pair<string, string>& img_pair, 
    const string& msg, bool write_to_file = true)
{
    std::cerr << "Error: " << msg << std::endl;
    if (write_to_file)
    {
        log << img_pair.first << "," << img_pair.second <<
            ",Error: " << msg << std::endl;
    } 
}

int main(int argc, char* argv[])
{
	// Parse command line arguments
	std::vector<string> source_paths, target_paths;
	std::vector<bool> toggle_source_seg, toggle_target_seg;
	std::vector<int> source_max_res, target_max_res;
    string seg_path, output_path, landmarks_path;
	string model_3dmm_h5_path, model_3dmm_dat_path;
	string reg_model_path, reg_deploy_path, reg_mean_path;
	string seg_model_path, seg_deploy_path;
    string log_path, cfg_path;
    bool generic, with_expr, with_gpu, reverse, cache, use_dlib;
    unsigned int gpu_device_id, verbose;
	try {
		options_description desc("Allowed options");
		desc.add_options()
			("help,h", "display the help message")
            ("verbose,v", value<unsigned int>(&verbose)->default_value(0), "output debug information [0, 2]")
			("sources", value<std::vector<string>>(&source_paths)->required(), "source paths")
			("targets,t", value<std::vector<string>>(&target_paths)->required(), "target paths")
			("source_seg", value<std::vector<bool>>(&toggle_source_seg), "toggle sources segmentation")
			("target_seg", value<std::vector<bool>>(&toggle_target_seg), "toggle targets segmentation")
			("source_max_res", value<std::vector<int>>(&source_max_res), "sources max resolution")
			("target_max_res", value<std::vector<int>>(&target_max_res), "targets max resolution")
			("reverse", value<bool>(&reverse)->default_value(false), "reverse swap direction")
			("output,o", value<string>(&output_path)->required(), "output directory")
            ("segmentations,s", value<string>(&seg_path)->default_value(""), "segmentations directory")
			("landmarks,l", value<string>(&landmarks_path)->required(), "path to landmarks model file")
			("model_3dmm_h5", value<string>(&model_3dmm_h5_path)->required(), "path to 3DMM file (.h5)")
			("model_3dmm_dat", value<string>(&model_3dmm_dat_path)->required(), "path to 3DMM file (.dat)")
			("reg_model,r", value<string>(&reg_model_path)->required(), "path to 3DMM regression CNN model file (.caffemodel)")
			("reg_deploy,d", value<string>(&reg_deploy_path)->required(), "path to 3DMM regression CNN deploy file (.prototxt)")
			("reg_mean,m", value<string>(&reg_mean_path)->required(), "path to 3DMM regression CNN mean file (.binaryproto)")
			("seg_model", value<string>(&seg_model_path), "path to face segmentation CNN model file (.caffemodel)")
			("seg_deploy", value<string>(&seg_deploy_path), "path to face segmentation CNN deploy file (.prototxt)")
            ("generic,g", value<bool>(&generic)->default_value(false), "use generic model without shape regression")
            ("expressions,e", value<bool>(&with_expr)->default_value(true), "with expressions")
			("cache,c", value<bool>(&cache)->default_value(false), "cache intermediate face data")
			("gpu", value<bool>(&with_gpu)->default_value(true), "toggle GPU / CPU")
			("gpu_id", value<unsigned int>(&gpu_device_id)->default_value(0), "GPU's device id")
            ("log", value<string>(&log_path)->default_value("face_swap_single2many_log.csv"), "log file path")
            ("cfg", value<string>(&cfg_path)->default_value("face_swap_single2many.cfg"), "configuration file (.cfg)")
            ("use_dlib", value<bool>(&use_dlib)->default_value(false), "use dlib")
			;
		variables_map vm;
		store(command_line_parser(argc, argv).options(desc).
			positional(positional_options_description().add("sources", -1)).run(), vm);

        if (vm.count("help")) {
            cout << "Usage: face_swap_single2many [options]" << endl;
            cout << desc << endl;
            exit(0);
        }

        // Read config file
        std::ifstream ifs(vm["cfg"].as<string>());
        store(parse_config_file(ifs, desc), vm);

        notify(vm);

		if (toggle_source_seg.empty()) toggle_source_seg.resize(source_paths.size(), true);
		if (toggle_target_seg.empty()) toggle_target_seg.resize(target_paths.size(), true);
		if (source_max_res.empty()) source_max_res.resize(source_paths.size(), 0);
		if (target_max_res.empty()) target_max_res.resize(target_paths.size(), 0);
		if(toggle_source_seg.size() != source_paths.size())
			throw error("Number of source_seg values must the same as the number of sources!");
		if (toggle_target_seg.size() != target_paths.size())
			throw error("Number of target_seg values must the same as the number of targets!");
		if (source_max_res.size() != source_paths.size())
			throw error("Number of source_max_res values must the same as the number of sources!");
		if (target_max_res.size() != target_paths.size())
			throw error("Number of target_max_res values must the same as the number of targets!");
        if(!seg_path.empty() && !is_directory(seg_path))
            throw error("segmentations must be a path to segmentations directory!");
        if ( !is_directory(output_path))
            throw error("output must be a path to a directory!");
		if (!is_regular_file(landmarks_path)) throw error("landmarks must be a path to a file!");
		if (!is_regular_file(model_3dmm_h5_path)) throw error("model_3dmm_h5 must be a path to a file!");
		if (!is_regular_file(model_3dmm_dat_path)) throw error("model_3dmm_dat must be a path to a file!");
		if (!is_regular_file(reg_model_path)) throw error("reg_model must be a path to a file!");
		if (!is_regular_file(reg_deploy_path)) throw error("reg_deploy must be a path to a file!");
		if (!is_regular_file(reg_mean_path)) throw error("reg_mean must be a path to a file!");
		if (!seg_model_path.empty() && !is_regular_file(seg_model_path))
			throw error("seg_model must be a path to a file!");
		if (!seg_deploy_path.empty() && !is_regular_file(seg_deploy_path))
			throw error("seg_deploy must be a path to a file!");
	}
	catch (const error& e) {
        cerr << "Error while parsing command-line arguments: " << e.what() << endl;
        cerr << "Use --help to display a list of options." << endl;
		exit(1);
	}

	try
	{
        // Initialize log file
        std::ofstream log;
        if (verbose > 0)
            log.open(log_path);

        // Parse image paths
		std::vector<string> src_img_paths, tgt_img_paths;
		std::vector<bool> toggle_src_img_seg, toggle_tgt_img_seg;
		std::vector<int> src_img_max_res, tgt_img_max_res;
		parseInputPaths(source_paths, src_img_paths,  toggle_source_seg,
			toggle_src_img_seg, source_max_res, src_img_max_res);
		parseInputPaths(target_paths, tgt_img_paths, toggle_target_seg,
			toggle_tgt_img_seg, target_max_res, tgt_img_max_res);

        // Initialize face swap
		std::shared_ptr<face_swap::FaceSwapEngine> fs =
			face_swap::FaceSwapEngine::createInstance(
				landmarks_path, model_3dmm_h5_path, model_3dmm_dat_path, reg_model_path,
				reg_deploy_path, reg_mean_path, seg_model_path, seg_deploy_path,
				generic, with_expr, with_gpu, gpu_device_id);

        // Initialize timer
        boost::timer::cpu_timer timer;
        float total_time = 0.0f, fps = 0.0f;
        int frame_counter = 0;

        // For each image pair
		for (size_t i = 0; i < src_img_paths.size(); ++i)
		{
			string& src_img_path = src_img_paths[i];
			bool toggle_src_seg = toggle_src_img_seg[i];
			int src_max_res = src_img_max_res[i];

			// Initialize source face data
			face_swap::FaceData src_face_data;
			if (!readFaceData(src_img_path, src_face_data))
			{
				src_face_data.enable_seg = toggle_src_seg;
				src_face_data.max_bbox_res = src_max_res;

				// Read source segmentations
				if (toggle_src_seg && seg_model_path.empty() && !seg_path.empty())
				{
					string src_seg_path = (path(seg_path) /=
						(path(src_img_path).stem() += ".png")).string();
					if (is_regular_file(src_seg_path))
						src_face_data.seg = cv::imread(src_seg_path, cv::IMREAD_GRAYSCALE);
				}
			}

			for (size_t j = 0; j < tgt_img_paths.size(); ++j)
			{
				string& tgt_img_path = tgt_img_paths[j];
				bool toggle_tgt_seg = toggle_tgt_img_seg[j];
				int tgt_max_res = tgt_img_max_res[j];

				if (src_img_path == tgt_img_path) continue;

				// Check if output image already exists
				path outputName = (path(src_img_path).stem() += "_") +=
					(path(tgt_img_path).stem() += ".jpg");
				string curr_output_path = (path(output_path) /= outputName).string();
				if (is_regular_file(curr_output_path))
				{
					std::cout << "Skipping: " << path(src_img_path).filename() <<
						" -> " << path(tgt_img_path).filename() << std::endl;
					continue;
				}
				std::cout << "Face swapping: " << path(src_img_path).filename() <<
					" -> " << path(tgt_img_path).filename() << std::endl;

				// Initialize target face data
				face_swap::FaceData tgt_face_data;
				if (!readFaceData(tgt_img_path, tgt_face_data))
				{
					tgt_face_data.enable_seg = toggle_tgt_seg;
					tgt_face_data.max_bbox_res = tgt_max_res;

					// Read target segmentations
					if (toggle_tgt_seg && seg_model_path.empty() && !seg_path.empty())
					{
						string tgt_seg_path = (path(seg_path) /=
							(path(tgt_img_path).stem() += ".png")).string();
						if (is_regular_file(tgt_seg_path))
							tgt_face_data.seg = cv::imread(tgt_seg_path, cv::IMREAD_GRAYSCALE);
					}
				}

				// Start measuring time
				timer.start();

				// Do face swap
				std::cout << "Processing source image..." << std::endl;
				if (!fs->process(src_face_data, cache, use_dlib))
				{
					logError(log, std::make_pair(src_img_path, tgt_img_path), "Failed to find a face in source image!", verbose);
					continue;
				}
				else if (cache)
					writeFaceData(src_img_path, src_face_data, false);

				std::cout << "Processing target image..." << std::endl;
				if(!fs->process(tgt_face_data, cache, use_dlib))
				{
					logError(log, std::make_pair(src_img_path, tgt_img_path), "Failed to find a face in target image!", verbose);
					continue;
				}
				else if (cache)
					writeFaceData(tgt_img_path, tgt_face_data, false);

				std::cout << "Swapping images..." << std::endl;
				cv::Mat rendered_img;
				if (!reverse) rendered_img = fs->swap(src_face_data, tgt_face_data, use_dlib);
				else rendered_img = fs->swap(tgt_face_data, src_face_data, use_dlib);
				if (rendered_img.empty())
				{
					logError(log, std::make_pair(src_img_path, tgt_img_path), "Face swap failed!", verbose);
					continue;
				}

				// Stop measuring time
				timer.stop();

				// Write output to file
				std::cout << "Writing " << outputName << " to output directory." << std::endl;
				cv::imwrite(curr_output_path, rendered_img);

				// Print current fps
				total_time += (timer.elapsed().wall*1.0e-9);
				fps = (++frame_counter) / total_time;
				std::cout << "total_time = " << total_time << std::endl;
				std::cout << "fps = " << fps << std::endl;

				// Debug
				if (verbose > 0)
				{
					// Write overlay image
					string debug_overlay_path = (path(output_path) /=
						(path(curr_output_path).stem() += "_overlay.jpg")).string();

					cv::Mat debug_result_img = rendered_img.clone();
					face_swap::renderImageOverlay(debug_result_img, tgt_face_data.scaled_bbox,
						src_face_data.cropped_img, tgt_face_data.cropped_img, cv::Scalar());
					cv::imwrite(debug_overlay_path, debug_result_img);
				}
				if (verbose > 1)
				{
					// Write rendered image
					string debug_render_path = (path(output_path) /=
						(path(curr_output_path).stem() += "_render.jpg")).string();

					cv::Mat src_render = fs->renderFaceData(src_face_data, 3.0f);
					cv::Mat tgt_render = fs->renderFaceData(tgt_face_data, 3.0f);
					cv::Mat debug_render_img;
					int width = std::min(src_render.cols, tgt_render.cols);
					if (src_render.cols > width)
					{
						int height = (int)std::round(src_render.rows * (float(width) / src_render.cols));
						cv::resize(src_render, src_render, cv::Size(width, height));
					}
					else
					{
						int height = (int)std::round(tgt_render.rows * (float(width) / tgt_render.cols));
						cv::resize(tgt_render, tgt_render, cv::Size(width, height));
					}
					cv::vconcat(src_render, tgt_render, debug_render_img);

					cv::imwrite(debug_render_path, debug_render_img);
				}
				//if (verbose > 1)
				//{
				//	// Write result image
				//	string debug_result_path = (path(output_path) /=
				//		(path(curr_output_path).stem() += "_result.jpg")).string();

				//	cv::Mat debug_result_img = face_swap::renderImagePipe(
				//	{ src_face_data.img, tgt_face_data.img, rendered_img });

				//	cv::imwrite(debug_result_path, debug_result_img);
				//}
			}
		} 
	}
	catch (std::exception& e)
	{
		cerr << e.what() << endl;
		return 1;
	}

	return 0;
}

