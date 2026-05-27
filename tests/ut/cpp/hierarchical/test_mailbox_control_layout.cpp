#include <gtest/gtest.h>

#include "worker_manager.h"

TEST(MailboxControlLayout, UsesFourArgsAndMovesResultToOffset48) {
    EXPECT_EQ(CTRL_OFF_ARG0, 16);
    EXPECT_EQ(CTRL_OFF_ARG1, 24);
    EXPECT_EQ(CTRL_OFF_ARG2, 32);
    EXPECT_EQ(CTRL_OFF_ARG3, 40);
    EXPECT_EQ(CTRL_OFF_RESULT, 48);
}
