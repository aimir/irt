from builtins import range
from numpy import (abs, array, concatenate, copy, exp, inf, log, log1p, max,
                   newaxis, sort, sum)
from scipy.optimize import minimize
from scipy.stats import dirichlet, lognorm, norm
from scipy.special import expit

__all__ = ['two_parameter_model', 'four_parameter_model', 'estimate_thetas']

OPTIMIZE_MAX_REACHED_MSG = ['Maximum number of iterations has been exceeded.',
                            'Maximum number of function evaluations '
                            'has been exceeded.']


def scale_guessing(value, c, d):
    return c + (d - c) * value


def two_parameter_model(a, b, theta):
    return expit(a * theta + b)


def four_parameter_model(a, b, c, d, theta):
    return scale_guessing(two_parameter_model(a, b, theta), c, d)


class StudentParametersDistribution(object):
    def __init__(self, theta_scale=1.):
        self.theta = norm(loc=0., scale=theta_scale)

    def rvs(self, size=None):
        return self.theta.rvs(size)

    def logpdf(self, theta):
        return self.theta.logpdf(theta)


class QuestionParametersDistribution(object):
    def __init__(self, a_scale=1.7, b_scale=1.,
                 c_d_dirichlet_alpha=(1, 1, 46)):
        self.a = lognorm(s=1., scale=a_scale)
        self.b = norm(scale=b_scale)
        self.c_d = dirichlet(alpha=c_d_dirichlet_alpha)

    def rvs(self, size=None):
        a = self.a.rvs(size)
        b = self.b.rvs(size)
        c_d = self.c_d.rvs(size)
        c = c_d[..., 0]
        d = 1 - c_d[..., 1]
        return concatenate((a[..., newaxis], b[..., newaxis],
                            c[..., newaxis], d[..., newaxis]), axis=-1)

    def logpdf(self, a, b, c, d):
        return (self.a.logpdf(a) + self.b.logpdf(b) +
                self.c_d.logpdf([c, 1 - d, d - c]))


def learn_abcd(thetas, question_dist, corrects):
    def f(arg):
        a, b, c, d = arg
        if a <= 0 or c < 0 or d > 1 or d - c < 0:
            return inf
        mult = question_dist.logpdf(a, b, c, d)
        for theta, correct in zip(thetas, corrects):
            p = four_parameter_model(a, b, c, d, theta)
            mult += log1p((2 * p - 1) * correct)
        return -mult
    return f


def learn_theta(abcds, student_dist, corrects):
    def f(theta):
        theta = theta[0]
        mult = student_dist.logpdf(theta)
        for abcd, correct in zip(abcds, corrects):
            a, b, c, d = abcd
            if correct != 0:
                p = four_parameter_model(a, b, c, d, theta)
                mult += log1p((2 * p - 1) * correct)
        return -mult
    return f


def expanded_scores(score_matrix):
    answers = [list(set(score)) for score in score_matrix]
    subscores = []

    for i, question_scores in enumerate(score_matrix):
        best = len(answers[i])
        subscores_per_student = []

        for student_score in question_scores:
            expanded = [1] * answers[i].index(student_score)
            if len(expanded) < best - 1:
                expanded = expanded + [-1]
                expanded = expanded + [0] * (best - 1 - len(expanded))
            subscores_per_student.append(expanded)

        subscores.append(array(subscores_per_student).T)
    return concatenate(subscores)


def initialize_random_values(students_count, subquestions_count,
                             student_dist, question_dist):
    theta_values = sum(student_dist.rvs(size=(100, students_count)),
                       axis=0) / 100
    abcd_values = sum(question_dist.rvs(size=(100, subquestions_count)),
                      axis=0) / 100
    return theta_values, abcd_values


def initialize_estimation(scores, student_dist, question_dist):
    # Even though we usually input the table as scores per student,
    # the analysis is easier for a table of scores per question:
    scores = scores.T
    questions_count, students_count = scores.shape
    answers_per_question = [sort(array(list(set(score))))
                            for score in scores]
    subquestions_per_question = [len(answers)
                                 for answers in answers_per_question]
    subquestions_count = sum(subquestions_per_question) - questions_count
    thetas, abcds = initialize_random_values(students_count,
                                             subquestions_count,
                                             student_dist, question_dist)
    expanded = expanded_scores(scores)
    return expanded, thetas, abcds


def parse_optimization_result(res):
    if not res['success'] and res['message'] not in OPTIMIZE_MAX_REACHED_MSG:
        raise RuntimeError("Optimization failed:\n" + repr(res))
    return res['x']


def question_abcd_given_theta(thetas, question_dist, scores, initial_abcd):
    to_minimize = learn_abcd(thetas, question_dist, scores)
    res = minimize(to_minimize, initial_abcd, method='Nelder-Mead')
    return parse_optimization_result(res)


def all_abcds_given_theta(thetas, question_dist, scores, abcds):
    return array([question_abcd_given_theta(thetas, question_dist,
                                            scores[i], abcds[i])
                  for i in range(len(abcds))])


def student_theta_given_abcd(abcds, student_dist, scores, inital_theta):
    to_minimize = learn_theta(abcds, student_dist, scores)
    res = minimize(to_minimize, [inital_theta], method='Nelder-Mead')
    return parse_optimization_result(res)


def all_thetas_given_abcd(abcds, student_dist, scores, thetas):
    return array([student_theta_given_abcd(abcds, student_dist,
                                           scores[:, i], thetas[i])
                  for i in range(len(thetas))])


def estimate_thetas(scores):
    student_dist = StudentParametersDistribution()
    question_dist = QuestionParametersDistribution()
    expanded, thetas, abcds = initialize_estimation(scores, student_dist,
                                                    question_dist)
    old_abcds, old_thetas = copy(abcds), copy(thetas)
    diff = 1
    small_diffs_streak = 0
    iter_count = 0
    while iter_count < 100 and small_diffs_streak < 3:
        old_abcds, old_thetas = copy(abcds), copy(thetas)
        abcds = all_abcds_given_theta(thetas, question_dist, expanded, abcds)
        thetas = all_thetas_given_abcd(abcds, student_dist, expanded, thetas)
        diff = max([max(abs(old_abcds - abcds)),
                    max(abs(old_thetas - thetas))])
        if diff < 0.001:
            small_diffs_streak += 1
        else:
            small_diffs_streak = 0
        iter_count += 1
        print diff
    return thetas, abcds
