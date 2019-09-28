


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

