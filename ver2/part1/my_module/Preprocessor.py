import json
import cv2
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Preprocessor:

    def __init__(self, meta_data, survey_original, survey_test, fixed_width=3 * 210, fixed_height=3 * 297):
        self.fixed_width = fixed_width
        self.fixed_height = fixed_height
        self.survey_original = cv2.imread(survey_original)
        self.survey_test = [cv2.imread(filename) for filename in survey_test]
        self.survey_original = self.img_resize(self.survey_original, ratio=0.99)
        self.survey_test = [self.img_resize(test, ratio=0.97) for test in self.survey_test]
        self.block_dict = dict()
        self.questions = dict()
        self.test_questions_list = list()
        self.query_square_list = list()
        self.question_square_list = list()
        self.questions_list = list()
        self.questions_text_list = list()
        self.parse_search_space(meta_data)

    def parse_search_space(self, filename):
        with open(filename, "r") as f:
            data = f.readlines()
        print(data)
        for line in data:
            line = line.strip()
            a_list = line.split(" ")
            idx = int(a_list[0])
            btype = int(a_list[1])
            if btype == 0 or btype == 3:
                option = None
            else:
                option = int(a_list[2])
            a_list = a_list[3].split(",")
            x = int(a_list[0])
            y = int(a_list[1])
            w = int(a_list[2]) - x
            h = int(a_list[3]) - y
            self.block_dict[idx] = {"type": btype, "option": option, "x": x, "y": y, "w": w, "h": h, "cnt": 0}
            if btype == 0 or btype == 3:
                self.questions[idx] = {"type": btype, "x": x, "y": y, "w": w, "h": h, "answers": []}
            else:
                self.questions[option]["answers"].append({"type": btype, "x": x, "y": y, "w": w, "h": h, "idx": idx})
        self.questions_list = [(k, v) for k, v in self.questions.items()]
        self.questions_list = sorted(self.questions_list, key=lambda d: d[0])

    def yfunc(self, xdata, img, ax):
        ydata = list()
        tmp_list = list()
        if ax == 1:
            tmp_list = img.T.copy()
        else:
            tmp_list = img.copy()

        for line in xdata:
            cnt = 0
            for px in tmp_list[line]:
                if px == 255:
                    cnt += 1
            ydata.append(cnt)
        return ydata

    def remove_border(self, img, ratio=0.97):
        """Removes white borders of an image

        :param img: image to remove white borders
        :param ratio: parameter to adjust noise
        :return:
        """
        height = img.shape[0]
        width = img.shape[1]
        left_bottom_x = 0
        left_bottom_y = 0
        w = 0
        h = 0

        # height cut
        xdata = np.arange(0, height, 1, int)
        ydata = self.yfunc(xdata, img, 0)

        for idx, y in enumerate(ydata):
            if y < width * ratio:
                left_bottom_x = idx
                break

        for idx, y in enumerate(reversed(ydata)):
            if y < width * ratio:
                w = height - idx + left_bottom_x
                break

        # width cut
        xdata = np.arange(0, width, 1, int)
        ydata = self.yfunc(xdata, img, 1)

        for idx, y in enumerate(ydata):
            if y < height * ratio:
                left_bottom_y = idx
                break

        for idx, y in enumerate(reversed(ydata)):
            if y < height * ratio:
                h = width - idx + left_bottom_y
                break

        print(left_bottom_x, left_bottom_y, w, h)
        return left_bottom_x, left_bottom_y, w, h

    def img_resize(self, img, ratio):
        """Removes white borders of an image and resizes it to self.fixed_height * self.fixed_weight

        :param img: image to be modified
        :return: modified image
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        x, y, w, h = self.remove_border(th, ratio=ratio)
        roi = img[x:x + w, y:y + h]
        cv2.namedWindow("win", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("win", self.fixed_width, self.fixed_height)
        cv2.imshow("win", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return cv2.resize(roi, (self.fixed_width, self.fixed_height), interpolation=cv2.INTER_NEAREST)

    def img_blending(self):
        """Shows blended image of the original survey and the test survey
        """
        cv2.namedWindow("dst", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("dst", self.fixed_width, self.fixed_height)
        for test in self.survey_test:
            dst = cv2.addWeighted(test, 0.7, self.survey_original, 0.3, 0)
            cv2.imshow("dst", dst)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def __displacement_fix_util(self, idx, th1, th2, ratio=0.4, use_white=False):
        """Adjust text block displacement between two images

        :param idx: block index number
        :param th1: binary threshold image of the original survey
        :param th2: binary threshold image of the test survey
        :param ratio: parameter to adjust noise
        :return: displacement delta x and delta y
        """
        w = self.block_dict[idx]["w"]
        h = self.block_dict[idx]["h"]
        x = self.block_dict[idx]["x"]
        y = self.block_dict[idx]["y"]

        delta_x = w // 2
        delta_y = h

        x_begin = max(0, x - delta_x)
        x_end = min(self.fixed_width, x + w + delta_x)
        y_begin = max(0, y - delta_y)
        y_end = min(self.fixed_height, y + h + delta_y)

        roi1 = th1[y:y+h, x: x + w]
        roi2 = th2[y_begin: y_end, x_begin:x_end]
        #cv2.imshow("1", roi1)
        #cv2.imshow("2", roi2)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        flag = True
        xpos = None
        ypos = None

        if not use_white:
            pixel_cnt = 0
            for xi in range(w):
                for yi in range(h):
                    if th1[y + yi, x + xi] == 0:
                        pixel_cnt += 1
            print("bcnt: {}".format(pixel_cnt))
        else:
            pixel_cnt = w*h

        for xpos in np.arange(x_begin, x_end, 1, int):
            for ypos in np.arange(y_begin, y_end, 1, int):
                cnt = 0
                flag = True
                for xdis in range(w):
                    for ydis in range(h):
                        if th1[y + ydis, x + xdis] == 255:
                            continue
                        if y + ydis >= self.fixed_height or ypos + ydis >= self.fixed_height or \
                                x + xdis >= self.fixed_width or xpos + xdis >= self.fixed_width:
                            continue
                        if th1[y + ydis, x + xdis] != th2[ypos + ydis, xpos + xdis]:
                            cnt += 1
                            if cnt > ratio*pixel_cnt:
                                flag = False
                                break
                    if flag is False:
                        break
                if flag:
                    break
            if flag:
                break

        x_begin = xpos
        x_end = xpos + w
        y_begin = ypos
        y_end = ypos + h
        roi2 = th2[y_begin: y_end, x_begin:x_end]
        #cv2.imshow("1", roi1)
        #cv2.imshow("2", roi2)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return xpos - x, ypos - y

    def draw_figure(self, x1, y1, x2, y2, w, h, th1, th2):
        plt.figure()

        # x coordinate
        plt.subplot(2, 1, 1)
        xdata1 = np.arange(x1, x1 + w, 1, int)
        func_data1 = self.yfunc(xdata1, th1, 0)
        plt.plot(xdata1, func_data1)

        xdata2 = np.arange(x2, x2 + w, 1, int)
        func_data2 = self.yfunc(xdata2, th2, 0)
        plt.plot(xdata2, func_data2)

        # y corrdinate
        plt.subplot(2, 1, 2)
        ydata1 = np.arange(y1, y1 + h, 1, int)
        func_data1 = self.yfunc(ydata1, th1, 1)
        plt.plot(ydata1, func_data1)

        ydata2 = np.arange(y2, y2 + h, 1, int)
        func_data2 = self.yfunc(ydata2, th2, 1)
        plt.plot(ydata2, func_data2)

        plt.show()

    def make_csv(self, csv_filename):
        max_sz = -1
        for k, v in self.questions.items():
            max_sz = max(max_sz, len(v["answers"]))
        a_dict = dict()
        for q_num, t in enumerate(sorted(self.questions.items(), key= lambda x: x[0])):
            k = t[0]
            v = t[1]
            if v["type"] == 3:
                a_dict[q_num] = -1
                continue
            a_dict[q_num] = [0 for i in range(max_sz)]
            for idx, item in enumerate(v["answers"]):
                a_dict[q_num][idx] = self.block_dict[item["idx"]]["cnt"]
        print(a_dict)
        df = pd.DataFrame(a_dict)
        df = df.T
        df["questions_text"] = pd.Series(self.questions_text_list)
        cols = list(df.columns.values)
        cols = cols[-1:] + cols[:-1]
        df = df[cols]
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')

    def print_answers(self, test_num):
        for q_idx, ele in enumerate(self.questions_list):
            v = ele[1]
            maxi_idx = -1
            maxi_q_num = -1
            maxi = -1
            if v["type"] == 3:
                continue
            for idx, item in enumerate(v["answers"]):
                q_num = item["idx"]
                tmp = self.test_questions_list[test_num][q_num]
                if tmp > maxi:
                    maxi = tmp
                    maxi_idx = idx
                    maxi_q_num = q_num
            self.block_dict[maxi_q_num]["cnt"] += 1
            print("problem {}, answer: {}".format(q_idx + 1, maxi_idx + 1))

    def classification(self, diff, img, test_num, q_num):
        image, contours, hierarchy = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        full_list = list()
        for cnt in contours:
            full_list.extend(cnt)
        full_list = np.array(full_list)
        #self.test_questions_list[test_num][q_num] = len(full_list)
        if len(full_list) == 0:
            return img
        x, y, w, h = cv2.boundingRect(full_list)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return img

    def noise_elimination(self, diff, img, test_num, q_num):
        kernel = np.ones((2, 2), np.uint8)
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=2)
        self.test_questions_list.append(dict())
        rec = self.classification(diff, img, test_num, q_num)
        #diff = cv2.dilate(diff, kernel, iterations=1)
        #_, diff = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY)
        return diff, rec

    def __check_position(self, idx, q_num, th1, th2, xdis, ydis, border=5):
        w = self.block_dict[q_num]["w"]
        h = self.block_dict[q_num]["h"]
        x = self.block_dict[q_num]["x"]
        y = self.block_dict[q_num]["y"]
        x2 = x + xdis
        y2 = y + ydis
        y1_begin = max(0, y - border)
        y1_end = min(self.fixed_height, y+h+border)
        x1_begin = max(0, x - border)
        x1_end = min(self.fixed_width, x + w + border)
        y2_begin = max(0, y2 - border)
        y2_end = min(self.fixed_height, y2+h+border)
        x2_begin = max(0, x2 - border)
        x2_end = min(self.fixed_width, x2 + w + border)


        roi1 = th1[max(0, y - border): min(self.fixed_height, y+h+border),
               max(0, x - border): min(self.fixed_width, x + w + border)]
        roi2 = th2[max(0, y2 - border): min(self.fixed_height, y2+h+border),
               max(0, x2 - border): min(self.fixed_width, x2 + w + border)]
        if roi1.shape != roi2.shape:
            roi2 = cv2.resize(roi2, (roi1.shape[1], roi1.shape[0]), interpolation=cv2.INTER_NEAREST)
        diff = cv2.subtract(roi1, roi2)
        diff, rec = self.noise_elimination(diff, self.survey_test[idx][y2_begin:y2_end, x2_begin: x2_end], idx, q_num)
        diff_width = diff.shape[1]
        diff_height = diff.shape[0]
        diff = diff[border: diff_height-border, border: diff_width-border]
        diff_width = diff.shape[1]
        diff_height = diff.shape[0]
        pixel_cnt = 0
        for i in range(diff_height):
            for j in range(diff_width):
                if diff[i, j] == 255:
                    pixel_cnt += 1
        self.test_questions_list[idx][q_num] = pixel_cnt
        cv2.imshow("ori", roi1)
        cv2.imshow("tst", roi2)
        cv2.imshow("diff", diff)
        cv2.imshow("draw_rec", rec)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __draw_rec2(self, idx, xdis, ydis):
        w = self.block_dict[idx]["w"]
        h = self.block_dict[idx]["h"]
        x = self.block_dict[idx]["x"]
        y = self.block_dict[idx]["y"]
        x2 = x + xdis
        y2 = y + ydis
        self.ori = cv2.rectangle(self.ori, (x, y), (x + w, y + h), (255, 0, 0), 2)
        self.ori = cv2.rectangle(self.ori, (x2, y2), (x2 + w, y2 + h), (0, 255, 0), 2)

    def displacement_fix(self):
        gray1 = cv2.cvtColor(self.survey_original, cv2.COLOR_BGR2GRAY)
        _, th1 = cv2.threshold(gray1, 200, 255, cv2.THRESH_BINARY)
        xdis = None
        ydis = None

        for idx, test in enumerate(self.survey_test):
            gray2 = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
            _, th2 = cv2.threshold(gray2, 200, 255, cv2.THRESH_BINARY)
            print("finished..{}", idx + 1)
            for k, v in self.block_dict.items():
                # Check whether or not if it's a question
                if v["type"] is 0 or v["type"] is 3:
                    xdis, ydis = self.__displacement_fix_util(k, th1, th2, ratio=0.45)
                #else:
                    #xdis, ydis = self.__displacement_fix_util(k, th1, th2, ratio=0.05, use_white=True)
                #self.__draw_rec2(k, xdis, ydis)
                self.__check_position(idx, k, th1, th2, xdis, ydis)
                #self.draw_figure(v["x"], v["y"], v["x"] + xdis, v["y"] + ydis, v["w"], v["h"], th1, th2)

    def overlapping_area(self, qs, ts, questions=True):
        if self.query_square_list[qs][0] > self.question_square_list[ts][1] or self.query_square_list[qs][1] < \
            self.question_square_list[ts][0] or self.query_square_list[qs][2] > self.question_square_list[ts][3] or \
                self.query_square_list[qs][3] < self.question_square_list[ts][2]:
            return -1
        tempx = [(0, 0), (0, 0)]
        tempy = [(0, 0), (0, 0)]
        tempx[0] = (self.query_square_list[qs][0], self.query_square_list[qs][1])
        tempx[1] = (self.question_square_list[ts][0], self.question_square_list[ts][1])
        tempy[0] = (self.query_square_list[qs][2], self.query_square_list[qs][3])
        tempy[1] = (self.question_square_list[ts][2], self.question_square_list[ts][3])
        w = tempx[0][1] - tempx[0][0]
        h = tempy[0][1] - tempy[0][0]
        area = w * h
        if tempx[0][0] > tempx[1][0]:
            tempx[0], tempx[1] = tempx[1], tempx[0]
        if tempy[0][0] > tempy[1][0]:
            tempy[0], tempy[1] = tempy[1], tempy[0]
        delx = tempx[0][1] - tempx[1][0] if tempx[0][1] < tempx[1][1] else tempx[1][1] - tempx[1][0]
        dely = tempy[0][1] - tempy[1][0] if tempy[0][1] < tempy[1][1] else tempy[1][1] - tempy[1][0]
        '''
        roi1 = self.survey_original[self.query_square_list[qs][2]: self.query_square_list[qs][3],
               self.query_square_list[qs][0]:self.query_square_list[qs][1]]
        roi2 = self.survey_original[self.question_square_list[ts][2]: self.question_square_list[ts][3],
               self.question_square_list[ts][0]:self.question_square_list[ts][1]]
        cv2.imshow("1", roi1)
        cv2.imshow("2", roi2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        if questions:
            if delx*dely > 0.5*area:
                return delx*dely
            else:
                return -1
        return delx * dely

    def load_original_survey_ocr(self, filepath):
        with open(filepath, "r", encoding="utf-8-sig") as f:
            js = f.read()

        data_list = json.loads(js)["data"]
        sentences = data_list[0]["text"].split("\n")
        data_list = data_list[1:]

        for idx, data in enumerate(data_list):
            yh = ast.literal_eval(data["bounds"][2])[1]
            yl = ast.literal_eval(data["bounds"][0])[1]
            xh = ast.literal_eval(data["bounds"][2])[0]
            xl = ast.literal_eval(data["bounds"][0])[0]
            self.query_square_list.append((xl, xh, yl, yh))

        iimg = self.survey_original.copy()
        for xx, yy, ww, hh in self.query_square_list:
            iimg = cv2.rectangle(iimg, (xx, yy), (xx + ww, yy + hh), (255, 0, 0), 2)
        for xx, yy, ww, hh in self.question_square_list:
            iimg = cv2.rectangle(iimg, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
        cv2.imshow("iimg", iimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        for ele in self.questions_list:
            v = ele[1]
            yh = v["y"] + v["h"]
            yl = v["y"]
            xh = v["x"] + v["w"]
            xl = v["x"]
            self.question_square_list.append((xl, xh, yl, yh))
        label = [-1] * len(self.query_square_list)
        for i in range(len(self.query_square_list)):
            maxi = 0
            for j in range(len(self.question_square_list)):
                temp = self.overlapping_area(i, j)
                if temp == -1:
                    continue
                if temp > maxi:
                    maxi = temp
                    label[i] = j



        word_bag = [None] * len(self.questions_list)
        for i in range(len(self.query_square_list)):
            j = label[i]
            if j == -1:
                continue
            else:
                if word_bag[j] is None:
                    word_bag[j] = list()
                word_bag[j].append(data_list[i]["text"])
        print(word_bag)
        sent_idx = 0
        bag_idx = 0

        while sent_idx < len(sentences) and bag_idx < len(word_bag):
            flag = True
            for word in word_bag[bag_idx]:
                if word not in sentences[sent_idx]:
                    flag = False
                    break
            if flag:
                self.questions_list[bag_idx][1]["question"] = sentences[sent_idx]
                bag_idx += 1
            sent_idx += 1

        for question in self.questions_list:
            self.questions_text_list.append(question[1]["question"])
