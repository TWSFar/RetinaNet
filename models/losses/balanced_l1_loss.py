        # balance L1 Loss
        a = 0.5
        b = math.e ** 3 - 1
        r = 1.5
        regression_loss = torch.where(
            torch.le(regression_diff, 1),
            a / b * (b * regression_diff + 1) *
            torch.log(b * regression_diff + 1) - a * regression_diff,
            r * regression_diff - a / b
        )