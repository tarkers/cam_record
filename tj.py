
import json
def write_2D_id_pose_json(or_eval=True):
       
    focus_ids = {}
    # get results json 
    f = open(rf"D:\Chen\cam_record\Test\JAPAN vs USA _ Highlights_OQT_17_21 2023\subject_2D\results.json")
    results = json.load(f)

    for item in results:
        new_data = {"image_id":item['image_id'],
                                    'keypoints':item['keypoints'],
                                    'box':item['box'],
                                    'score':item['score']}
        if item['idx'] not in focus_ids:
            focus_ids[item['idx']] = [new_data]
        else:
            focus_ids[item['idx']].append(new_data)
            
    print(focus_ids.keys())
    
if __name__ == '__main__':
    write_2D_id_pose_json()