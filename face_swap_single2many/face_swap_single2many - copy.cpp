
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

	if (src_img_path == tgt_img_path) return;

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

	// Do face swap
	std::cout << "Processing source image..." << std::endl;
	if (!fs->process(src_face_data, cache, use_dlib))
	{
		logError(log, std::make_pair(src_img_path, tgt_img_path), "Failed to find a face in source image!", verbose);
		return;
	}
	else if (cache)
		writeFaceData(src_img_path, src_face_data, false);

	std::cout << "Processing target image..." << std::endl;
	if(!fs->process(tgt_face_data, cache, use_dlib))
	{
		logError(log, std::make_pair(src_img_path, tgt_img_path), "Failed to find a face in target image!", verbose);
		return;
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
		return;
	}		

	// Write output to file
	std::cout << "Writing " << outputName << " to output directory." << std::endl;
	cv::imwrite(curr_output_path, rendered_img);
	
	return;
