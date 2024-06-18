def execute_script_with_modified_args(script_path, args_dict):
    with open(script_path, 'r', encoding='utf-8') as file:
        script_content = file.read()

    # argparse 부분을 대체하기 위한 코드 준비
    args_code = f"path_to_img = '{args_dict['image']}'\npath_to_txt = 'class0518/classes.txt'\npath_to_class = '{args_dict['classes']}'\npath_to_annotaitons = 'class0518/annotations'\n"
    
    # args 변수를 전역 네임스페이스에 추가
    globals()['args'] = args_dict

    # 원본 스크립트에서 argparse 부분을 대체할 코드로 변경
    modified_script_content = script_content.replace('args = vars(ap.parse_args())', args_code)

    # 변경된 스크립트 실행
    exec(modified_script_content, globals())

# 첫 번째 실행에 필요한 인자
first_args = {
    "image": "class0518/processtrain/images",
    "txt": "class0518/processtrain/labels",
    "classes": "class0518/classes.txt"
}

# 두 번째 실행에 필요한 인자
second_args = {
    "image": "class0518/valid/images",
    "txt": "class0518/valid/labels",
    "classes": "class0518/classes.txt"
}

# yolo_to_json.py 스크립트 실행
script_path = "yolo_to_json.py"
execute_script_with_modified_args(script_path, first_args)
execute_script_with_modified_args(script_path, second_args)