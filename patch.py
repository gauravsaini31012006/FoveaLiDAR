import re

with open('_process_with_foveation.py', 'r', encoding='utf-8') as f:
    content = f.read()

pattern = re.compile(r'            if frame_idx % 2 == 0:.*?(?=            frame_gen.generate_frame)', re.DOTALL)
replacement = '''            if frame_idx % 2 == 0:
                gaze_points = gaze_sim.simulate_driving_gaze(preprocessed_pc)
            else:
                gaze_points = gaze_sim.simulate_saliency_gaze(preprocessed_pc)

            roi_def.rois.clear()
            if len(gaze_points) > 0:
                primary_gaze = gaze_points[0]
                roi_def.add_roi((float(primary_gaze[0]), float(primary_gaze[1]), float(primary_gaze[2])))
            else:
                import numpy as np
                primary_gaze = np.zeros(3, dtype=np.float32)
                roi_def.add_roi((0.0, 0.0, 0.0))

            compressed_pc = compression.compress_adaptive(preprocessed_pc, strategy="voxel")

            roi_ids, _ = roi_def.classify_points(compressed_pc)
            bev_image = renderer.render_scifi_3d(compressed_pc, roi_ids, primary_gaze, frame_idx)
'''

content = pattern.sub(replacement, content)
with open('_process_with_foveation.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Patched successfully!')
